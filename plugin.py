"""GPT-SoVITS 语音回复插件。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
from urllib.parse import urljoin

import asyncio
import base64
import copy
import hashlib

import httpx
from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase, Tool
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

MediaType = Literal["wav", "raw", "ogg", "aac"]

_CONTROL_SUPPRESS_DEFAULT = 0
_SUPPORTED_MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}


class PluginSectionConfig(PluginConfigBase):
    """插件基础配置。"""

    __ui_label__ = "插件"
    __ui_icon__ = "package"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用插件。")
    config_version: str = Field(default="1.1.1", description="配置版本。")
    debug: bool = Field(default=False, description="是否输出调试日志。")


class ApiConfig(PluginConfigBase):
    """GPT-SoVITS API 配置。"""

    __ui_label__ = "API"
    __ui_icon__ = "server"
    __ui_order__ = 1

    api_base_url: str = Field(default="http://127.0.0.1:9880", description="GPT-SoVITS API 根地址。")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="请求超时时间（秒）。")
    request_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="额外请求头，可用于 Authorization、X-API-Key 等鉴权信息。",
    )


class WeightConfig(PluginConfigBase):
    """GPT-SoVITS 权重配置。"""

    __ui_label__ = "权重"
    __ui_icon__ = "sliders-horizontal"
    __ui_order__ = 2

    apply_weights_on_load: bool = Field(default=True, description="插件加载或配置更新时是否应用权重路径。")
    require_weights_ready: bool = Field(default=False, description="权重应用失败时是否禁止后续 TTS。")
    gpt_weights_path: str = Field(default="", description="GPT 权重路径，留空则不切换。")
    sovits_weights_path: str = Field(default="", description="SoVITS 权重路径，留空则不切换。")


class TTSConfig(PluginConfigBase):
    """GPT-SoVITS /tts 基础请求配置。"""

    __ui_label__ = "合成"
    __ui_icon__ = "audio-lines"
    __ui_order__ = 3

    text_lang: str = Field(default="zh", description="待合成文本语言。")
    ref_audio_path: str = Field(default="", description="参考音频路径。")
    aux_ref_audio_paths: List[str] = Field(default_factory=list, description="辅助参考音频路径列表。")
    prompt_lang: str = Field(default="zh", description="参考音频提示词语言。")
    prompt_text: str = Field(default="", description="参考音频对应提示词。")
    media_type: MediaType = Field(default="wav", description="输出音频格式：wav/raw/ogg/aac。")


class InferenceConfig(PluginConfigBase):
    """GPT-SoVITS 推理参数配置。"""

    __ui_label__ = "推理"
    __ui_icon__ = "settings-2"
    __ui_order__ = 4

    enabled: bool = Field(default=False, description="是否向 /tts 请求写入本节推理参数；关闭时使用 GPT-SoVITS API 默认值。")
    top_k: int = Field(default=5, ge=1, le=100, description="top_k 采样参数。")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="top_p 采样参数。")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="采样温度。")
    text_split_method: str = Field(default="cut5", description="文本切分方法。")
    batch_size: int = Field(default=1, ge=1, le=128, description="批处理大小。")
    batch_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="批处理阈值。")
    split_bucket: bool = Field(default=True, description="是否启用分桶推理。")
    speed_factor: float = Field(default=1.0, ge=0.25, le=4.0, description="语速倍率。")
    fragment_interval: float = Field(default=0.3, ge=0.0, le=10.0, description="分片间隔。")
    seed: int = Field(default=-1, description="随机种子，-1 表示随机。")
    parallel_infer: bool = Field(default=True, description="是否并行推理。")
    repetition_penalty: float = Field(default=1.35, ge=0.0, le=10.0, description="重复惩罚。")
    sample_steps: int = Field(default=32, ge=1, le=128, description="采样步数。")
    super_sampling: bool = Field(default=False, description="是否启用超采样。")
    extra_tts_params: Dict[str, Any] = Field(default_factory=dict, description="未来兼容的额外 /tts 参数。")


class BehaviorConfig(PluginConfigBase):
    """语音化行为配置。"""

    __ui_label__ = "行为"
    __ui_icon__ = "message-square"
    __ui_order__ = 5

    session_mode_enabled: bool = Field(default=True, description="是否允许通过命令开启会话级持续语音模式。")
    allow_group: bool = Field(default=True, description="是否允许群聊使用。")
    allow_private: bool = Field(default=True, description="是否允许私聊使用。")
    allowed_group_ids: List[str] = Field(default_factory=list, description="允许使用的群 ID，空列表表示不限制。")
    control_user_ids: List[str] = Field(default_factory=list, description="允许执行 /voice 命令的用户 ID，空列表表示不限制。")
    fallback_to_text_on_error: bool = Field(default=True, description="TTS 失败时是否回退发送原文本。")
    max_text_length: int = Field(default=300, ge=1, le=5000, description="允许语音化的最大文本长度。")


class GPTSoVITSVoiceReplyConfig(PluginConfigBase):
    """GPT-SoVITS 语音回复插件配置。"""

    plugin: PluginSectionConfig = Field(default_factory=PluginSectionConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    weights: WeightConfig = Field(default_factory=WeightConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)


@dataclass(slots=True)
class VoiceSessionState:
    """单个会话的语音化状态。"""

    enabled: bool = False
    once_pending: bool = False
    control_ack_suppress_count: int = _CONTROL_SUPPRESS_DEFAULT


class GPTSoVITSRequestError(RuntimeError):
    """GPT-SoVITS API 请求失败。"""


class GPTSoVITSClient:
    """GPT-SoVITS API 客户端。"""

    def __init__(self, logger: Any) -> None:
        """初始化客户端。"""

        self._logger = logger

    @staticmethod
    def normalize_base_url(api_base_url: str) -> str:
        """规范化 API 根地址。"""

        normalized = str(api_base_url or "").strip()
        if not normalized:
            raise GPTSoVITSRequestError("GPT-SoVITS API 地址不能为空")
        return normalized.rstrip("/") + "/"

    @staticmethod
    def build_url(api_base_url: str, endpoint: str) -> str:
        """构造 API 地址。"""

        return urljoin(GPTSoVITSClient.normalize_base_url(api_base_url), endpoint.lstrip("/"))

    @staticmethod
    def sanitize_headers(raw_headers: Dict[str, Any]) -> Dict[str, str]:
        """校验并规范化请求头。"""

        sanitized_headers: Dict[str, str] = {}
        for raw_key, raw_value in raw_headers.items():
            key = str(raw_key or "").strip()
            value = str(raw_value or "")
            if not key:
                raise GPTSoVITSRequestError("request_headers 存在空请求头名称")
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise GPTSoVITSRequestError(f"request_headers.{key} 包含非法换行字符")
            if not value:
                continue
            sanitized_headers[key] = value
        return sanitized_headers

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        """从错误响应中提取可读错误。"""

        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                for key in ("message", "detail", "error"):
                    value = payload.get(key)
                    if value:
                        return str(value)
        text = response.text.strip()
        return text[:500] if text else f"HTTP {response.status_code}"

    async def apply_weights(self, config: GPTSoVITSVoiceReplyConfig, applied_weights: Tuple[str, str]) -> Tuple[str, str]:
        """按配置切换 GPT 与 SoVITS 权重。"""

        if not config.weights.apply_weights_on_load:
            return applied_weights

        target_gpt = config.weights.gpt_weights_path.strip()
        target_sovits = config.weights.sovits_weights_path.strip()
        current_gpt, current_sovits = applied_weights

        headers = self.sanitize_headers(config.api.request_headers)
        timeout = httpx.Timeout(config.api.timeout_seconds)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            if target_gpt and target_gpt != current_gpt:
                await self._get_json(
                    client,
                    self.build_url(config.api.api_base_url, "/set_gpt_weights"),
                    {"weights_path": target_gpt},
                )
                current_gpt = target_gpt
                self._logger.info("已应用 GPT-SoVITS GPT 权重配置。")

            if target_sovits and target_sovits != current_sovits:
                await self._get_json(
                    client,
                    self.build_url(config.api.api_base_url, "/set_sovits_weights"),
                    {"weights_path": target_sovits},
                )
                current_sovits = target_sovits
                self._logger.info("已应用 GPT-SoVITS SoVITS 权重配置。")

        return current_gpt, current_sovits

    async def synthesize(self, text: str, config: GPTSoVITSVoiceReplyConfig) -> bytes:
        """调用 /tts 合成完整音频。"""

        payload = self.build_tts_payload(text, config)
        headers = self.sanitize_headers(config.api.request_headers)
        timeout = httpx.Timeout(config.api.timeout_seconds)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            response = await client.post(self.build_url(config.api.api_base_url, "/tts"), json=payload)

        if response.status_code < 200 or response.status_code >= 300:
            detail = self._extract_error_detail(response)
            raise GPTSoVITSRequestError(f"GPT-SoVITS /tts 请求失败: status={response.status_code} detail={detail}")

        audio_bytes = response.content
        if not audio_bytes:
            raise GPTSoVITSRequestError("GPT-SoVITS /tts 返回了空音频")
        return audio_bytes

    @staticmethod
    async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> None:
        """发送权重切换 GET 请求并检查响应状态。"""

        response = await client.get(url, params=params)
        if response.status_code < 200 or response.status_code >= 300:
            detail = GPTSoVITSClient._extract_error_detail(response)
            raise GPTSoVITSRequestError(f"GPT-SoVITS 权重切换失败: status={response.status_code} detail={detail}")

    @staticmethod
    def build_tts_payload(text: str, config: GPTSoVITSVoiceReplyConfig) -> Dict[str, Any]:
        """构造 GPT-SoVITS /tts 请求体。"""

        media_type = str(config.tts.media_type or "wav").strip().lower()
        if media_type not in _SUPPORTED_MEDIA_TYPES:
            raise GPTSoVITSRequestError(f"不支持的 GPT-SoVITS 输出格式: {media_type}")

        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": config.tts.text_lang.strip(),
            "ref_audio_path": config.tts.ref_audio_path.strip(),
            "aux_ref_audio_paths": [item.strip() for item in config.tts.aux_ref_audio_paths if item.strip()],
            "prompt_text": config.tts.prompt_text,
            "prompt_lang": config.tts.prompt_lang.strip(),
            "media_type": media_type,
            "streaming_mode": False,
        }

        if config.inference.enabled:
            payload.update(
                {
                    "top_k": config.inference.top_k,
                    "top_p": config.inference.top_p,
                    "temperature": config.inference.temperature,
                    "text_split_method": config.inference.text_split_method.strip(),
                    "batch_size": config.inference.batch_size,
                    "batch_threshold": config.inference.batch_threshold,
                    "split_bucket": config.inference.split_bucket,
                    "speed_factor": config.inference.speed_factor,
                    "fragment_interval": config.inference.fragment_interval,
                    "seed": config.inference.seed,
                    "parallel_infer": config.inference.parallel_infer,
                    "repetition_penalty": config.inference.repetition_penalty,
                    "sample_steps": config.inference.sample_steps,
                    "super_sampling": config.inference.super_sampling,
                }
            )

            for key, value in config.inference.extra_tts_params.items():
                normalized_key = str(key or "").strip()
                if not normalized_key:
                    continue
                if normalized_key == "streaming_mode":
                    continue
                payload[normalized_key] = value

        return payload


class GPTSoVITSVoiceReplyPlugin(MaiBotPlugin):
    """GPT-SoVITS 语音回复插件。"""

    config_model = GPTSoVITSVoiceReplyConfig

    def __init__(self) -> None:
        """初始化插件运行时状态。"""

        super().__init__()
        self._state_lock = asyncio.Lock()
        self._session_states: Dict[str, VoiceSessionState] = defaultdict(VoiceSessionState)
        self._client = GPTSoVITSClient(self._get_logger())
        self._applied_weights: Tuple[str, str] = ("", "")
        self._weights_ready = True

    async def on_load(self) -> None:
        """处理插件加载。"""

        self._client = GPTSoVITSClient(self.ctx.logger)
        await self._apply_weights_from_config()
        self.ctx.logger.info("GPT-SoVITS 语音回复插件已加载。")

    async def on_unload(self) -> None:
        """处理插件卸载。"""

        async with self._state_lock:
            self._session_states.clear()
        self.ctx.logger.info("GPT-SoVITS 语音回复插件已卸载。")

    async def on_config_update(self, scope: str, config_data: Dict[str, object], version: str) -> None:
        """处理插件配置热重载。"""

        del config_data
        self.ctx.logger.info(f"GPT-SoVITS 语音回复插件配置已更新: scope={scope}, version={version}")
        await self._apply_weights_from_config()

    def get_components(self) -> List[Dict[str, Any]]:
        """收集组件并拉平 Maisaka 工具可见性元数据。"""

        components = super().get_components()
        for component in components:
            if component.get("name") != "use_voice_reply":
                continue
            metadata = component.get("metadata")
            if not isinstance(metadata, dict):
                continue
            nested_metadata = metadata.get("metadata")
            if isinstance(nested_metadata, dict):
                for key in ("visibility", "core_tool"):
                    if key in nested_metadata:
                        metadata[key] = nested_metadata[key]
        return components

    def _get_active_config(self) -> GPTSoVITSVoiceReplyConfig:
        """获取当前生效配置，未注入时回退默认配置。"""

        try:
            return cast(GPTSoVITSVoiceReplyConfig, self.config)
        except RuntimeError:
            return GPTSoVITSVoiceReplyConfig()

    async def _apply_weights_from_config(self) -> None:
        """按配置应用 GPT-SoVITS 权重。"""

        config = self._get_active_config()
        if not config.plugin.enabled:
            self._weights_ready = True
            return

        try:
            self._applied_weights = await self._client.apply_weights(config, self._applied_weights)
            self._weights_ready = True
        except Exception as exc:
            self._weights_ready = False
            self._get_logger().error(f"应用 GPT-SoVITS 权重失败: {exc}")

    @staticmethod
    def _build_hook_result(modified_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """构造 Hook 继续执行结果。"""

        return {"action": "continue", "modified_kwargs": modified_kwargs}

    @staticmethod
    def _build_abort_result(reason: str, modified_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """构造 Hook 中止结果。"""

        return {"action": "abort", "reason": reason, "modified_kwargs": modified_kwargs}

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        """规范化会话 ID。"""

        return str(session_id or "").strip()

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        """规范化用户 ID。"""

        return str(user_id or "").strip()

    @staticmethod
    def _is_group_message(message: Dict[str, Any]) -> bool:
        """判断消息是否属于群聊。"""

        message_info = message.get("message_info")
        if not isinstance(message_info, dict):
            return False
        return bool(message_info.get("group_info"))

    @classmethod
    def _extract_group_id(cls, message: Dict[str, Any]) -> str:
        """从消息中提取群 ID。"""

        message_info = message.get("message_info")
        if not isinstance(message_info, dict):
            return ""
        group_info = message_info.get("group_info")
        if not isinstance(group_info, dict):
            return ""
        return str(group_info.get("group_id") or "").strip()

    @classmethod
    def _is_chat_allowed(cls, message: Dict[str, Any], config: GPTSoVITSVoiceReplyConfig) -> Tuple[bool, str]:
        """判断当前聊天是否允许语音化。"""

        if cls._is_group_message(message):
            if not config.behavior.allow_group:
                return False, "当前配置不允许群聊使用语音回复"
            allowed_group_ids = {str(item).strip() for item in config.behavior.allowed_group_ids if str(item).strip()}
            group_id = cls._extract_group_id(message)
            if allowed_group_ids and group_id not in allowed_group_ids:
                return False, f"当前群不在语音回复白名单中: group_id={group_id}"
            return True, ""

        if not config.behavior.allow_private:
            return False, "当前配置不允许私聊使用语音回复"
        return True, ""

    @staticmethod
    def _check_control_permission(user_id: str, config: GPTSoVITSVoiceReplyConfig) -> bool:
        """检查 /voice 命令权限。"""

        allowed_user_ids = {str(item).strip() for item in config.behavior.control_user_ids if str(item).strip()}
        return not allowed_user_ids or user_id in allowed_user_ids

    @staticmethod
    def _contains_only_convertible_segments(raw_message: Any) -> bool:
        """判断消息段是否仅包含可保留或可转换的类型。"""

        if not isinstance(raw_message, list) or not raw_message:
            return False
        allowed_types = {"text", "reply"}
        for segment in raw_message:
            if not isinstance(segment, dict):
                return False
            segment_type = str(segment.get("type") or "").strip().lower()
            if segment_type not in allowed_types:
                return False
        return True

    @staticmethod
    def _extract_text(raw_message: Any, processed_plain_text: str = "") -> str:
        """从可转换消息段中提取文本。"""

        text_parts: List[str] = []
        if isinstance(raw_message, list):
            for segment in raw_message:
                if not isinstance(segment, dict):
                    continue
                if str(segment.get("type") or "").strip().lower() != "text":
                    continue
                data = segment.get("data")
                if isinstance(data, str) and data.strip():
                    text_parts.append(data.strip())
        text = "\n".join(text_parts).strip()
        if text:
            return text
        return processed_plain_text.strip()

    @staticmethod
    def _build_voice_segment(audio_bytes: bytes, text: str) -> Dict[str, Any]:
        """构造插件运行时 voice 消息段。"""

        return {
            "type": "voice",
            "data": "[语音消息]",
            "hash": hashlib.sha256(audio_bytes).hexdigest(),
            "binary_data_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "source_text": text,
        }

    @classmethod
    def _replace_text_with_voice(cls, message: Dict[str, Any], audio_bytes: bytes, text: str) -> Dict[str, Any]:
        """将文本消息替换为语音消息，同时保留回复组件。"""

        modified_message = copy.deepcopy(message)
        raw_message = modified_message.get("raw_message")
        preserved_segments: List[Dict[str, Any]] = []
        if isinstance(raw_message, list):
            for segment in raw_message:
                if not isinstance(segment, dict):
                    continue
                if str(segment.get("type") or "").strip().lower() == "reply":
                    preserved_segments.append(copy.deepcopy(segment))
        preserved_segments.append(cls._build_voice_segment(audio_bytes, text))
        modified_message["raw_message"] = preserved_segments
        modified_message["processed_plain_text"] = "[语音消息]"
        modified_message["is_emoji"] = False
        modified_message["is_picture"] = False
        modified_message["is_command"] = False
        return modified_message

    async def _get_state_snapshot(self, session_id: str) -> VoiceSessionState:
        """读取会话状态快照。"""

        async with self._state_lock:
            state = self._session_states[session_id]
            return VoiceSessionState(
                enabled=state.enabled,
                once_pending=state.once_pending,
                control_ack_suppress_count=state.control_ack_suppress_count,
            )

    async def _set_session_enabled(self, session_id: str, enabled: bool) -> None:
        """设置会话级语音回复状态。"""

        async with self._state_lock:
            state = self._session_states[session_id]
            state.enabled = enabled
            if not enabled:
                state.once_pending = False

    async def _set_once_pending(self, session_id: str) -> None:
        """设置下一条回复语音化。"""

        async with self._state_lock:
            self._session_states[session_id].once_pending = True

    async def _increase_control_suppress_count(self, session_id: str) -> None:
        """标记下一条控制命令回执不做语音化。"""

        async with self._state_lock:
            self._session_states[session_id].control_ack_suppress_count += 1

    async def _consume_control_suppress(self, session_id: str) -> bool:
        """消费控制命令回执抑制计数。"""

        async with self._state_lock:
            state = self._session_states[session_id]
            if state.control_ack_suppress_count <= 0:
                return False
            state.control_ack_suppress_count -= 1
            return True

    async def _should_voice_reply(self, session_id: str) -> bool:
        """判断并消费一次性语音化状态。"""

        async with self._state_lock:
            state = self._session_states[session_id]
            if state.once_pending:
                state.once_pending = False
                return True
            return state.enabled

    @Command(
        "voice_reply",
        description="控制 GPT-SoVITS 语音回复模式",
        pattern=r"^/voice(?:\s+(?P<action>on|off|once|status))?\s*$",
        intercept_message_level=1,
    )
    async def handle_voice_command(
        self,
        stream_id: str = "",
        user_id: str = "",
        matched_groups: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, bool]:
        """处理 /voice 控制命令。"""

        del kwargs

        config = self._get_active_config()
        if not config.plugin.enabled:
            return False, "语音回复插件未启用。", True
        if not stream_id:
            return False, "无法获取当前聊天流。", True
        if not self._check_control_permission(user_id, config):
            await self._increase_control_suppress_count(stream_id)
            await self.ctx.send.text("你没有权限控制语音回复。", stream_id)
            return False, "没有权限", True

        action = str((matched_groups or {}).get("action") or "status").strip().lower()
        if action == "on":
            if not config.behavior.session_mode_enabled:
                await self._increase_control_suppress_count(stream_id)
                await self.ctx.send.text("当前配置不允许开启会话级持续语音模式。", stream_id)
                return False, "会话级语音模式未启用", True
            await self._set_session_enabled(stream_id, True)
            await self._increase_control_suppress_count(stream_id)
            await self.ctx.send.text("已开启本会话语音回复。", stream_id)
            return True, "已开启本会话语音回复", True

        if action == "off":
            await self._set_session_enabled(stream_id, False)
            await self._increase_control_suppress_count(stream_id)
            await self.ctx.send.text("已关闭本会话语音回复。", stream_id)
            return True, "已关闭本会话语音回复", True

        if action == "once":
            await self._set_once_pending(stream_id)
            await self._increase_control_suppress_count(stream_id)
            await self.ctx.send.text("下一条回复将使用语音。", stream_id)
            return True, "下一条回复将使用语音", True

        state = await self._get_state_snapshot(stream_id)
        status_text = "开启" if state.enabled else "关闭"
        once_text = "有" if state.once_pending else "无"
        await self._increase_control_suppress_count(stream_id)
        await self.ctx.send.text(f"语音回复状态：持续模式={status_text}，下一条语音={once_text}。", stream_id)
        return True, "已查询语音回复状态", True

    @Tool(
        "use_voice_reply",
        description="让下一条可见回复使用 GPT-SoVITS 语音发送。只设置状态，不直接发送消息。",
        parameters={
            "reason": {
                "type": "string",
                "description": "选择语音回复的原因。",
                "default": "",
            }
        },
        visibility="visible",
        core_tool=True,
    )
    async def handle_use_voice_reply(
        self,
        stream_id: str = "",
        chat_id: str = "",
        reason: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """标记当前会话下一条回复使用语音。"""

        del kwargs

        config = self._get_active_config()
        if not config.plugin.enabled:
            return {"success": False, "content": "语音回复插件未启用。"}

        target_stream_id = self._normalize_session_id(stream_id or chat_id)
        if not target_stream_id:
            return {"success": False, "content": "无法确定当前聊天流，未设置语音回复。"}

        await self._set_once_pending(target_stream_id)
        content = "下一条可见回复将使用语音。"
        if reason.strip() and config.plugin.debug:
            self._get_logger().debug(f"use_voice_reply 已标记会话: stream_id={target_stream_id} reason={reason}")
        return {"success": True, "content": content}

    @HookHandler(
        "send_service.before_send",
        name="gpt_sovits_voice_before_send",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=60000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_before_send(
        self,
        hook_name: str = "",
        message: Any = None,
        typing: bool = False,
        set_reply: bool = False,
        reply_message_id: Optional[str] = None,
        storage_message: bool = True,
        show_log: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """在发送前按需将文本回复改写为语音回复。"""

        del hook_name

        modified_kwargs = dict(kwargs)
        modified_kwargs.update(
            {
                "message": message,
                "typing": typing,
                "set_reply": set_reply,
                "reply_message_id": reply_message_id,
                "storage_message": storage_message,
                "show_log": show_log,
            }
        )

        config = self._get_active_config()
        if not config.plugin.enabled:
            return self._build_hook_result(modified_kwargs)
        if not isinstance(message, dict):
            return self._build_hook_result(modified_kwargs)

        session_id = self._normalize_session_id(str(message.get("session_id") or ""))
        if not session_id:
            return self._build_hook_result(modified_kwargs)

        if await self._consume_control_suppress(session_id):
            return self._build_hook_result(modified_kwargs)

        should_voice = await self._should_voice_reply(session_id)
        if not should_voice:
            return self._build_hook_result(modified_kwargs)

        allowed, reason = self._is_chat_allowed(message, config)
        if not allowed:
            self._get_logger().info(f"跳过语音化: session_id={session_id} reason={reason}")
            return self._build_hook_result(modified_kwargs)

        if config.weights.require_weights_ready and not self._weights_ready:
            error_message = "GPT-SoVITS 权重尚未成功应用，已跳过语音化。"
            self._get_logger().error(error_message)
            if config.behavior.fallback_to_text_on_error:
                return self._build_hook_result(modified_kwargs)
            return self._build_abort_result(error_message, modified_kwargs)

        raw_message = message.get("raw_message")
        if not self._contains_only_convertible_segments(raw_message):
            self._get_logger().info(f"跳过语音化: session_id={session_id} reason=消息包含不可转换的非文本段")
            return self._build_hook_result(modified_kwargs)

        text = self._extract_text(raw_message, str(message.get("processed_plain_text") or ""))
        if not text:
            self._get_logger().info(f"跳过语音化: session_id={session_id} reason=未提取到文本")
            return self._build_hook_result(modified_kwargs)
        if len(text) > config.behavior.max_text_length:
            self._get_logger().warning(
                f"跳过语音化: session_id={session_id} reason=文本过长 length={len(text)} limit={config.behavior.max_text_length}"
            )
            return self._build_hook_result(modified_kwargs)

        try:
            audio_bytes = await self._client.synthesize(text, config)
        except Exception as exc:
            self._get_logger().error(f"GPT-SoVITS 语音合成失败: session_id={session_id} error={exc}")
            if config.behavior.fallback_to_text_on_error:
                return self._build_hook_result(modified_kwargs)
            return self._build_abort_result("GPT-SoVITS 语音合成失败", modified_kwargs)

        modified_kwargs["message"] = self._replace_text_with_voice(message, audio_bytes, text)
        if config.plugin.debug:
            self._get_logger().debug(f"已将文本回复转换为语音: session_id={session_id} bytes={len(audio_bytes)}")
        return self._build_hook_result(modified_kwargs)


def create_plugin() -> GPTSoVITSVoiceReplyPlugin:
    """创建插件实例。"""

    return GPTSoVITSVoiceReplyPlugin()
