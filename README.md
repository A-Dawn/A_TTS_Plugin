# A_TTS_plugin

一款文字转语音插件，目前只支持GPT_Sovits推理。

## 功能

- `/voice on`：本会话持续语音回复。
- `/voice off`：关闭本会话语音回复。
- `/voice once`：下一条回复使用语音。
- `/voice status`：查看当前状态。
- `use_voice_reply`：供 Maisaka 决策调用，只标记下一条回复需要语音化。

插件只生成 MaiBot 标准 `voice` 消息段，所以理论上应该支持Snowluma，Napcat或者其他写好语音兼容的适配器...大概？

## 配置重点

- `api.api_base_url`：GPT-SoVITS API 地址，例如 `http://127.0.0.1:9880`。
- `api.request_headers`：可选鉴权请求头，例如 `Authorization` 或 `X-API-Key`。日志不会输出请求头值。
- `weights.gpt_weights_path` / `weights.sovits_weights_path`：需要切换的权重路径。
- `tts` 与 `inference`：默认使用 GPT-SoVITS API 推理参数；将 `inference.enabled` 改为 `true` 后才会发送本地推理配置。
- `behavior.fallback_to_text_on_error`：语音合成失败时是否发送原文本。

完整字段说明见 [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)。

## 注意

默认不启用语音模式。请先在 WebUI 插件配置中启用插件，并确认 GPT-SoVITS API 可访问。
