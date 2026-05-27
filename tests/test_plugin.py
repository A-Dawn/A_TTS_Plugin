"""GPT-SoVITS 语音回复插件测试。"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import base64
import importlib.util
import sys

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = PLUGIN_ROOT / "plugin.py"


def _load_plugin_module() -> Any:
    """按文件路径加载插件模块，避免插件目录必须是 Python 包。"""

    module_name = "_test_gpt_sovits_voice_reply_plugin"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 GPT-SoVITS 语音回复插件模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


plugin_module = _load_plugin_module()


class _FakeLogger:
    """记录插件日志，便于断言错误与跳过原因。"""

    def __init__(self) -> None:
        self.records: Dict[str, List[str]] = {"debug": [], "info": [], "warning": [], "error": []}

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.records["debug"].append(str(message))

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.records["info"].append(str(message))

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.records["warning"].append(str(message))

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.records["error"].append(str(message))


class _FakeSend:
    """最小发送能力假对象。"""

    def __init__(self) -> None:
        self.text_calls: List[Tuple[str, str]] = []

    async def text(self, text: str, stream_id: str) -> bool:
        self.text_calls.append((text, stream_id))
        return True


class _FakeTTSClient:
    """可控的 GPT-SoVITS 客户端假对象。"""

    def __init__(self, audio_bytes: bytes = b"voice-bytes", error: Optional[Exception] = None) -> None:
        self.audio_bytes = audio_bytes
        self.error = error
        self.calls: List[Tuple[str, Any]] = []

    async def synthesize(self, text: str, config: Any) -> bytes:
        self.calls.append((text, config))
        if self.error is not None:
            raise self.error
        return self.audio_bytes


def _build_config() -> Any:
    """构造默认启用的插件配置。"""

    config = plugin_module.GPTSoVITSVoiceReplyConfig()
    config.plugin.enabled = True
    config.api.api_base_url = "http://tts.local:9880"
    config.tts.ref_audio_path = "ref.wav"
    config.tts.prompt_text = "你好"
    return config


def _build_plugin(config: Any | None = None) -> Any:
    """构造已注入配置和上下文的插件实例。"""

    plugin = plugin_module.GPTSoVITSVoiceReplyPlugin()
    logger = _FakeLogger()
    send = _FakeSend()
    plugin._set_context(SimpleNamespace(logger=logger, send=send))
    plugin.set_plugin_config((config or _build_config()).model_dump(mode="python"))
    return plugin


def _build_message(raw_message: List[Dict[str, Any]], group_id: str = "") -> Dict[str, Any]:
    """构造 send_service.before_send 使用的消息字典。"""

    group_info = {"group_id": group_id, "group_name": "测试群"} if group_id else None
    return {
        "session_id": "stream-1",
        "message_id": "message-1",
        "platform": "qq",
        "message_info": {
            "user_info": {"user_id": "user-1", "user_nickname": "测试用户"},
            "group_info": group_info,
        },
        "processed_plain_text": "你好，世界",
        "raw_message": raw_message,
    }


def test_sanitize_headers_rejects_empty_key_and_newline_value() -> None:
    headers = plugin_module.GPTSoVITSClient.sanitize_headers(
        {
            "Authorization": "Bearer token",
            "X-API-Key": "secret",
            "X-Empty": "",
        }
    )

    assert headers == {"Authorization": "Bearer token", "X-API-Key": "secret"}

    with pytest.raises(plugin_module.GPTSoVITSRequestError, match="空请求头名称"):
        plugin_module.GPTSoVITSClient.sanitize_headers({"": "token"})

    with pytest.raises(plugin_module.GPTSoVITSRequestError, match="非法换行字符"):
        plugin_module.GPTSoVITSClient.sanitize_headers({"Authorization": "Bearer token\nbad"})


def test_build_tts_payload_parameterizes_request_and_forces_non_streaming() -> None:
    config = _build_config()
    config.inference.enabled = True
    config.tts.text_lang = "zh"
    config.tts.aux_ref_audio_paths = [" aux-1.wav ", "", "aux-2.wav"]
    config.tts.media_type = "ogg"
    config.inference.top_k = 12
    config.inference.top_p = 0.8
    config.inference.temperature = 0.7
    config.inference.speed_factor = 1.2
    config.inference.extra_tts_params = {"streaming_mode": True, "custom_flag": "ok"}

    payload = plugin_module.GPTSoVITSClient.build_tts_payload("需要语音化的文本", config)

    assert payload["text"] == "需要语音化的文本"
    assert payload["text_lang"] == "zh"
    assert payload["ref_audio_path"] == "ref.wav"
    assert payload["aux_ref_audio_paths"] == ["aux-1.wav", "aux-2.wav"]
    assert payload["media_type"] == "ogg"
    assert payload["top_k"] == 12
    assert payload["top_p"] == 0.8
    assert payload["temperature"] == 0.7
    assert payload["speed_factor"] == 1.2
    assert payload["custom_flag"] == "ok"
    assert payload["streaming_mode"] is False


def test_build_tts_payload_can_use_server_default_inference_params() -> None:
    config = _build_config()
    config.inference.enabled = False
    config.inference.top_k = 12
    config.inference.extra_tts_params = {"streaming_mode": True, "custom_flag": "ok"}

    payload = plugin_module.GPTSoVITSClient.build_tts_payload("使用服务端默认推理参数", config)

    assert payload["text"] == "使用服务端默认推理参数"
    assert payload["media_type"] == "wav"
    assert payload["streaming_mode"] is False
    assert "top_k" not in payload
    assert "custom_flag" not in payload


@pytest.mark.asyncio
async def test_apply_weights_gets_changed_weights_with_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()
    config.api.request_headers = {"Authorization": "Bearer test-token"}
    config.weights.gpt_weights_path = "/models/gpt.ckpt"
    config.weights.sovits_weights_path = "/models/sovits.pth"

    responses = [
        plugin_module.httpx.Response(200, content=b"ok"),
        plugin_module.httpx.Response(200, content=b"ok"),
    ]

    class FakeAsyncClient:
        instances: List["FakeAsyncClient"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.gets: List[Tuple[str, Dict[str, Any]]] = []
            FakeAsyncClient.instances.append(self)

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback

        async def get(self, url: str, params: Dict[str, Any]) -> Any:
            self.gets.append((url, params))
            return responses.pop(0)

    monkeypatch.setattr(plugin_module.httpx, "AsyncClient", FakeAsyncClient)

    client = plugin_module.GPTSoVITSClient(_FakeLogger())
    applied_weights = await client.apply_weights(config, ("", ""))

    instance = FakeAsyncClient.instances[0]
    assert instance.kwargs["headers"] == {"Authorization": "Bearer test-token"}
    assert applied_weights == ("/models/gpt.ckpt", "/models/sovits.pth")
    assert instance.gets == [
        ("http://tts.local:9880/set_gpt_weights", {"weights_path": "/models/gpt.ckpt"}),
        ("http://tts.local:9880/set_sovits_weights", {"weights_path": "/models/sovits.pth"}),
    ]


@pytest.mark.asyncio
async def test_synthesize_success_and_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_config()
    responses = [
        plugin_module.httpx.Response(200, content=b"audio"),
        plugin_module.httpx.Response(500, content=b'{"message":"boom"}', headers={"content-type": "application/json"}),
        plugin_module.httpx.Response(200, content=b""),
    ]

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
            del exc_type, exc, traceback

        async def post(self, url: str, json: Dict[str, Any]) -> Any:
            del url, json
            return responses.pop(0)

    monkeypatch.setattr(plugin_module.httpx, "AsyncClient", FakeAsyncClient)

    client = plugin_module.GPTSoVITSClient(_FakeLogger())
    assert await client.synthesize("你好", config) == b"audio"

    with pytest.raises(plugin_module.GPTSoVITSRequestError, match="boom"):
        await client.synthesize("你好", config)

    with pytest.raises(plugin_module.GPTSoVITSRequestError, match="空音频"):
        await client.synthesize("你好", config)


@pytest.mark.asyncio
async def test_before_send_converts_text_to_voice_and_preserves_context() -> None:
    plugin = _build_plugin()
    plugin._client = _FakeTTSClient(audio_bytes=b"audio-bytes")
    await plugin._set_once_pending("stream-1")

    message = _build_message(
        [
            {"type": "reply", "data": {"target_message_id": "origin-1"}},
            {"type": "text", "data": "你好"},
            {"type": "text", "data": "世界"},
        ]
    )

    result = await plugin.handle_before_send(message=message, set_reply=True, reply_message_id="origin-1")
    modified_message = result["modified_kwargs"]["message"]

    assert result["action"] == "continue"
    assert modified_message["session_id"] == "stream-1"
    assert modified_message["message_id"] == "message-1"
    assert result["modified_kwargs"]["set_reply"] is True
    assert result["modified_kwargs"]["reply_message_id"] == "origin-1"
    assert modified_message["processed_plain_text"] == "[语音消息]"
    assert modified_message["raw_message"][0]["type"] == "reply"
    assert modified_message["raw_message"][1]["type"] == "voice"
    assert base64.b64decode(modified_message["raw_message"][1]["binary_data_base64"]) == b"audio-bytes"
    assert plugin._client.calls[0][0] == "你好\n世界"


@pytest.mark.asyncio
async def test_before_send_aborts_on_tts_failure_when_fallback_disabled() -> None:
    config = _build_config()
    config.behavior.fallback_to_text_on_error = False
    plugin = _build_plugin(config)
    plugin._client = _FakeTTSClient(error=plugin_module.GPTSoVITSRequestError("failed"))
    await plugin._set_once_pending("stream-1")

    result = await plugin.handle_before_send(message=_build_message([{"type": "text", "data": "你好"}]))

    assert result["action"] == "abort"
    assert result["reason"] == "GPT-SoVITS 语音合成失败"


@pytest.mark.asyncio
async def test_before_send_skips_mixed_message_and_consumes_once() -> None:
    plugin = _build_plugin()
    plugin._client = _FakeTTSClient(audio_bytes=b"audio")
    await plugin._set_once_pending("stream-1")
    message = _build_message(
        [
            {"type": "text", "data": "你好"},
            {"type": "image", "data": "base64://image"},
        ]
    )

    result = await plugin.handle_before_send(message=message)
    state = await plugin._get_state_snapshot("stream-1")

    assert result["action"] == "continue"
    assert result["modified_kwargs"]["message"] == message
    assert plugin._client.calls == []
    assert state.once_pending is False


@pytest.mark.asyncio
async def test_voice_command_sets_state_and_suppresses_control_ack() -> None:
    plugin = _build_plugin()
    plugin._client = _FakeTTSClient(error=AssertionError("控制回执不应触发 TTS"))

    handled, message, intercepted = await plugin.handle_voice_command(
        stream_id="stream-1",
        user_id="user-1",
        matched_groups={"action": "on"},
    )
    state = await plugin._get_state_snapshot("stream-1")
    hook_result = await plugin.handle_before_send(message=_build_message([{"type": "text", "data": "控制回执"}]))
    state_after_hook = await plugin._get_state_snapshot("stream-1")

    assert handled is True
    assert message == "已开启本会话语音回复"
    assert intercepted is True
    assert plugin.ctx.send.text_calls == [("已开启本会话语音回复。", "stream-1")]
    assert state.enabled is True
    assert state.control_ack_suppress_count == 1
    assert hook_result["modified_kwargs"]["message"]["processed_plain_text"] == "你好，世界"
    assert state_after_hook.enabled is True
    assert state_after_hook.control_ack_suppress_count == 0


@pytest.mark.asyncio
async def test_use_voice_reply_tool_is_visible_and_marks_once() -> None:
    plugin = _build_plugin()

    tool_components = [component for component in plugin.get_components() if component.get("name") == "use_voice_reply"]
    result = await plugin.handle_use_voice_reply(stream_id="stream-1", reason="适合语音")
    state = await plugin._get_state_snapshot("stream-1")

    assert len(tool_components) == 1
    assert tool_components[0]["metadata"]["visibility"] == "visible"
    assert tool_components[0]["metadata"]["core_tool"] is True
    assert result == {"success": True, "content": "下一条可见回复将使用语音。"}
    assert state.once_pending is True
