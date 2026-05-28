# A_TTS_plugin 配置字段说明

本文档解释 `config.toml` 中每个配置项的含义、默认值和使用建议。

## plugin

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | 是否启用插件。关闭时命令、工具和发送前 Hook 都不会进行语音化处理。 |
| `config_version` | `"1.1.3"` | 配置结构版本号。运行时用它判断是否需要按默认配置补齐字段。 |
| `debug` | `false` | 是否输出调试日志。开启后会记录工具触发、语音转换成功等更多信息，但不会打印请求头值。 |

## api

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `api_base_url` | `"http://127.0.0.1:9880"` | GPT-SoVITS API 根地址。插件会访问 `/tts`、`/set_gpt_weights`、`/set_sovits_weights`。 |
| `timeout_seconds` | `30.0` | GPT-SoVITS 请求超时时间，单位为秒。生成长文本或首次加载模型较慢时可以适当调大。 |

## api.request_headers

用于给所有 GPT-SoVITS 请求附加请求头，常用于鉴权。

```toml
[api.request_headers]
Authorization = "Bearer your-token"
X-API-Key = "your-api-key"
```

注意事项：

- 请求头 key 不能为空。
- 请求头 key/value 不能包含换行字符。
- 空 value 会被忽略。
- 插件日志不会输出请求头 value。

## weights

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `apply_weights_on_load` | `true` | 插件加载或配置更新时是否应用权重路径。 |
| `require_weights_ready` | `false` | 权重切换失败后是否禁止后续 TTS。设为 `true` 时，权重未成功应用会按失败策略处理。 |
| `gpt_weights_path` | `""` | GPT 权重路径。留空时不调用 `/set_gpt_weights`。 |
| `sovits_weights_path` | `""` | SoVITS 权重路径。留空时不调用 `/set_sovits_weights`。 |

权重切换只在插件加载或配置更新时执行。普通 TTS 请求不会重复切换权重，避免每条回复额外阻塞。

## tts

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `text_lang` | `"zh"` | 待合成文本语言。常见值取决于 GPT-SoVITS 服务端支持情况，例如 `zh`、`en`、`ja`。 |
| `ref_audio_path` | `""` | 参考音频路径。通常需要填写 GPT-SoVITS 服务端可访问的本地路径。 |
| `aux_ref_audio_paths` | `[]` | 辅助参考音频路径列表。用于多参考音频场景，服务端不需要时保持空列表。 |
| `prompt_lang` | `"zh"` | 参考音频提示词语言。 |
| `prompt_text` | `""` | 参考音频对应的提示词文本。 |
| `media_type` | `"wav"` | 输出音频格式。当前插件允许 `wav`、`raw`、`ogg`、`aac`。 |

插件始终向 `/tts` 发送 `streaming_mode = false`，最终发送给 MaiBot 的是一整段音频 bytes。

## inference

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | 是否向 `/tts` 请求写入本节推理参数。默认关闭，使用 GPT-SoVITS API 服务端默认推理参数。 |
| `top_k` | `5` | top-k 采样参数。仅 `enabled = true` 时发送。 |
| `top_p` | `1.0` | top-p 采样参数。仅 `enabled = true` 时发送。 |
| `temperature` | `1.0` | 采样温度。值越高通常随机性越强。仅 `enabled = true` 时发送。 |
| `text_split_method` | `"cut5"` | 文本切分方法。具体可用值以 GPT-SoVITS 服务端为准。仅 `enabled = true` 时发送。 |
| `batch_size` | `1` | 批处理大小。仅 `enabled = true` 时发送。 |
| `batch_threshold` | `0.75` | 批处理阈值。仅 `enabled = true` 时发送。 |
| `split_bucket` | `true` | 是否启用分桶推理。仅 `enabled = true` 时发送。 |
| `speed_factor` | `1.0` | 语速倍率。仅 `enabled = true` 时发送。 |
| `fragment_interval` | `0.3` | 分片间隔。仅 `enabled = true` 时发送。 |
| `seed` | `-1` | 随机种子，`-1` 表示由服务端随机。仅 `enabled = true` 时发送。 |
| `parallel_infer` | `true` | 是否并行推理。仅 `enabled = true` 时发送。 |
| `repetition_penalty` | `1.35` | 重复惩罚参数。仅 `enabled = true` 时发送。 |
| `sample_steps` | `32` | 采样步数。仅 `enabled = true` 时发送。 |
| `super_sampling` | `false` | 是否启用超采样。仅 `enabled = true` 时发送。 |

默认建议保持 `enabled = false`。当你确认服务端默认推理效果不满足需求，或需要固定音色稳定性、语速、随机种子时，再开启并调整本节参数。

## inference.extra_tts_params

用于兼容未来 GPT-SoVITS `/tts` 新增参数，只有 `inference.enabled = true` 时才会发送。

```toml
[inference.extra_tts_params]
custom_param = "value"
```

注意：插件会强制忽略这里的 `streaming_mode`，避免误开启流式输出。

## behavior

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `session_mode_enabled` | `true` | 是否允许通过 `/voice on` 开启会话级持续语音模式。 |
| `allow_group` | `true` | 是否允许群聊使用语音回复。 |
| `allow_private` | `true` | 是否允许私聊使用语音回复。 |
| `allowed_group_ids` | `[]` | 群聊白名单。空列表表示不限制；填写后只有列表中的群 ID 可以使用。 |
| `control_user_ids` | `[]` | `/voice` 命令控制者白名单。空列表表示不限制；填写后只有列表中的用户 ID 可以控制语音模式。 |
| `admin_user_ids` | `[]` | `/voice on/off <群号或 QQ 号>` 管理员白名单。留空时复用 `control_user_ids`；两者都为空则没有用户可执行目标聊天开关。 |
| `fallback_to_text_on_error` | `true` | TTS 失败时是否发送原文本。设为 `false` 时，语音合成失败会中止本次发送。 |
| `max_text_length` | `300` | 允许语音化的最大文本长度。超过后跳过语音化并发送原文本。 |

`/voice on 123456789`、`/voice off 123456789` 会按目标 ID 控制语音回复。
目标 ID 可以是群号，也可以是私聊 QQ 号，插件不会区分两者。

## 常见配置片段

只启用基础语音回复，使用服务端默认推理参数：

```toml
[plugin]
enabled = true

[tts]
ref_audio_path = "D:/GPT-SoVITS/ref.wav"
prompt_text = "这是一段参考音频文本"
```

启用鉴权请求头：

```toml
[api.request_headers]
Authorization = "Bearer your-token"
```

只允许指定群使用：

```toml
[behavior]
allow_group = true
allowed_group_ids = ["123456789"]
```

需要手动控制推理参数：

```toml
[inference]
enabled = true
speed_factor = 1.1
seed = 1234
```
