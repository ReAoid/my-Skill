---
name: 视频总结工具
description: >
  视频批量转写为TXT文本。当用户提到"视频转文字"、"获取字幕"、"字幕提取"时触发。
  支持单个视频URL、UP主空间URL、本地音频、本地视频，自动检测字幕（优先提取），无字幕或本地文件时使用 ASR 转写。
  输出结构化 TXT 文件（元信息+带时间戳原文），供AI后续加工。
---

# 视频总结工具

将视频内容批量转写为结构化 TXT 文件，用于后续 AI 分析或知识管理。

---

## 核心流程

```
用户输入（URL / 本地音频 / 本地视频）
  ↓
1. 解析意图：单个视频 / UP主批量 / 本地媒体文件
  ↓
2. 网络视频先检查字幕：yt-dlp --list-subs
   ├── 有字幕 → 直接提取（秒级完成）
   └── 无字幕 → 下载原始音频 → Whisper 转写（实时进度报告）
   本地视频 → ffmpeg 无损提取音轨 → Whisper 转写
  ↓
3. 输出结构化 TXT（元信息 + 完整文本 + 带时间戳分段）
```

---

## 依赖项

执行前必须确认以下依赖已安装：

| 依赖 | 安装命令 | 必需 |
|------|---------|------|
| yt-dlp | `pip install yt-dlp` | 是 |
| mlx-whisper | `pip install mlx-whisper` | 是（Apple Silicon）/ 否（Windows） |
| faster-whisper | `pip install faster-whisper` | 是（Windows）/ 备选（macOS） |
| ffmpeg | 本地便携版或系统安装 | 是 |
| opencc | `pip install opencc-python-reimplemented` | 推荐 |
| torch (CUDA) | 参见踩坑经验 | 推荐（Windows GPU） |
| funasr / modelscope / soundfile / torchaudio | `python3 -m pip install --user funasr modelscope soundfile torchaudio` | 备用（中文识别） |

### 环境检查命令

```bash
yt-dlp --version
python -c "import mlx_whisper; print('mlx-whisper: OK')"  # Apple Silicon
python -c "from faster_whisper import WhisperModel; print('OK')"  # Windows/备选
python -c "from funasr import AutoModel; print('FunASR: OK')"  # 中文备用
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
ffmpeg -version
```

### macOS (Apple Silicon) 注意事项

- **优先使用 mlx-whisper**：faster-whisper 的 ctranslate2 不支持 Apple GPU，只能跑 CPU，极慢
- mlx-whisper 使用 Metal GPU 加速，28分钟视频约3-4分钟完成
- 模型仓库：`mlx-community/whisper-small-mlx`、`mlx-community/whisper-medium-mlx`
- 首次使用需从 HuggingFace 下载模型

---

## 工作流程详解

### Step 1：解析用户输入

- **单个视频**：`https://www.bilibili.com/video/BV1xxxxxx/`
- **UP主空间**：`https://space.bilibili.com/123456`
- **BV号**：`BV1xxxxxx`（自动补全为完整URL）
- **本地音频**：`.m4a` / `.mp3` / `.wav` / `.aac` / `.flac` / `.opus` / `.ogg` / `.webm`
- **本地视频**：`.mp4` / `.mov` / `.mkv` / `.avi` / `.flv` / `.wmv` / `.m4v` / `.webm`

本地视频不要先转成 mp3。优先用 ffmpeg 直接复制音轨到 m4a：

```bash
ffmpeg -y -i "input.mp4" -vn -c:a copy "output.m4a"
```

这样不会重编码，速度通常比 `-c:a aac` 或转 mp3 快很多，也能降低长视频被截断的风险。只有 `-c:a copy` 失败时，才用 AAC 转码兜底。

如果是UP主空间，先用 `yt-dlp --flat-playlist` 获取视频列表。

**重要**：
- 默认不限制处理数量（处理全部视频）
- 视频超过20个时，**先提示用户确认**，告知预计耗时
- 支持用户指定范围，如"最新的10个"、"前5个"

### Step 2：字幕优先策略

对每个视频先检查是否有可用字幕：

```bash
yt-dlp --list-subs "视频URL"
```

- 如果有 `zh-CN` / `zh-Hans` / `zh` 字幕 → 直接提取：
  ```bash
  yt-dlp --write-subs --sub-langs zh-CN --skip-download -o "输出路径" "视频URL"
  ```
  然后将 `.srt` 或 `.json` 字幕文件转为 Markdown

- 如果只有弹幕（danmaku）或无字幕 → 走下载+转写路线

### Step 3：下载或提取音频（无字幕/本地视频时）

网络视频保留原始音频格式，不主动转 mp3：

```bash
yt-dlp -x --audio-quality 0 -o "输出路径" "视频URL"
```

本地视频直接交给脚本，脚本会自动提取音频并转写：

```bash
python -u <skill_dir>/scripts/bilibili_pipeline.py "本地视频.mp4"
```

### Step 4：Faster Whisper 转写

使用脚本执行转写，**实时报告进度**（每5秒输出：百分比、已处理时间、片段数、预计剩余时间）：

```bash
python <skill_dir>/scripts/bilibili_pipeline.py --transcribe-only "音频路径"
```

默认配置：
- 模型：`small`
- 量化：`int8`
- 设备：`cuda`（不可用时回退到 `cpu`）
- 语言：`zh`

### Step 4b：FunASR 备用转写（Apple Silicon 默认 MPS）

默认仍使用 Whisper / mlx-whisper。需要中文备用识别时显式加 `--engine funasr`：

```bash
python3 -u <skill_dir>/scripts/bilibili_pipeline.py --transcribe-only "音频路径" --engine funasr
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine auto
```

FunASR 默认模型为 `paraformer-zh`，并启用 `fsmn-vad` 与 `ct-punc-c`。`--funasr-device auto` 会优先使用 CUDA；Apple Silicon 上优先使用 `mps`；不可用时回退 CPU。首次运行会从 ModelScope 下载模型到 `~/.cache/modelscope/`，之后可复用本地缓存。

可手动指定设备：

```bash
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr --funasr-device mps
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr --funasr-device cpu
```

进度输出示例：
```
[进度]  24.4% | 已处理 87s / 356s | 片段数: 28, 预计剩余 0分46秒
[进度] 100.0% | 转写完成 | 用时 0分49秒 | 共 126 个片段
```

### Step 5：输出结构化 TXT

每个视频生成一个 TXT 文件，格式如下：

```
TITLE: 视频标题
ID: BV1xxxxxx
URL: https://www.bilibili.com/video/BV1xxxxxx/
UPLOADER: xxx
DATE: 2024-01-01
DURATION: 35分20秒
METHOD: Faster Whisper (small, int8)
PROCESSED_AT: 2024-01-01 12:00:00
SEGMENT_COUNT: 120

=== FULL TEXT ===
（完整转写文本，无时间戳，供AI阅读分析）

=== TIMESTAMPS ===
[00:00 -> 00:15] 第一段文字
[00:15 -> 00:30] 第二段文字
...

=== END ===
```

**设计说明**：
- 顶部元信息用 `KEY: VALUE` 格式，便于程序解析
- `FULL TEXT` 区提供纯文本全文，供 AI 直接阅读
- `TIMESTAMPS` 区提供带时间戳的分段，便于定位原始内容
- 不由脚本生成 Markdown，交由 AI 根据需要自行加工

---

## 输出目录

默认输出到当前工作目录下的 `bilibili_output/`：

```
bilibili_output/
├── BV1xxxxxx/
│   ├── 视频标题.m4a      # 音频文件（可选保留，优先原始格式）
│   └── 视频标题.txt       # 转写 TXT
├── BV2xxxxxx/
│   └── 视频标题.txt       # 有字幕时无音频文件
├── progress.json          # 断点续传进度
└── ...
```

---

## 断点续传

在输出目录下维护一个 `progress.json` 文件：

```json
{
  "processed": ["BV1xxxxxx", "BV2xxxxxx"],
  "failed": ["BV3xxxxxx"],
  "last_update": "2024-01-01T12:00:00"
}
```

批量处理时跳过已处理的视频，中断后可继续。

---

## 调用方式（重要）

**不要使用 `2>&1` 重定向**，直接运行即可，`-u` 参数禁用 Python 缓冲：

```powershell
python -u "<skill_dir>/scripts/bilibili_pipeline.py" "视频URL"
```

原因：PowerShell 的 `2>&1` 会将 stderr 编码为 CLIXML 格式，导致输出解析失败。脚本已内置 `sys.stderr = sys.stdout`，所有错误自动走 stdout。

## 脚本调用方式

```bash
# 处理单个视频
python -u <skill_dir>/scripts/bilibili_pipeline.py "https://www.bilibili.com/video/BV1xxxxxx/"

# 批量处理UP主视频（全部）
python -u <skill_dir>/scripts/bilibili_pipeline.py "https://space.bilibili.com/123456"

# 批量处理UP主视频（前10个）
python -u <skill_dir>/scripts/bilibili_pipeline.py "https://space.bilibili.com/123456" --limit 10

# 指定输出目录
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --output-dir "./my_output"

# 指定模型
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --model base

# 使用 FunASR 中文备用识别
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr

# Apple Silicon 上显式使用 MPS
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr --funasr-device mps

# Whisper 失败时自动回退 FunASR
python3 -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine auto

# 不保留音频文件
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --cleanup-audio

# 仅转写已有音频
python -u <skill_dir>/scripts/bilibili_pipeline.py --transcribe-only "音频路径"

# 处理本地视频：自动无损提取音轨并转写
python -u <skill_dir>/scripts/bilibili_pipeline.py "本地视频.mp4"

# 仅从本地视频提取音频，不转写
python -u <skill_dir>/scripts/bilibili_pipeline.py --extract-audio "本地视频.mp4" --output-dir "./输出目录"

# 指定本地视频/音频输出目录
python -u <skill_dir>/scripts/bilibili_pipeline.py "本地视频.mp4" --output-dir "./输出目录"
```

---

## 本地视频转写实战经验

这次处理 73分24秒本地 mp4 的经验：

- 源视频音轨本身是 AAC，最优方式是 `ffmpeg -vn -c:a copy` 直接抽成 `.m4a`，实际约 38 秒完成；误用 AAC 重编码会慢很多。
- 不要为了“通用”先转 mp3。mp3 转码多一步损耗，长视频还有截断和耗时风险；Whisper 可以直接吃 `.m4a`。
- Apple Silicon 上优先用 `mlx-whisper`，`small` 模型转写 73 分钟音频约 11 分钟，输出 2577 个片段，速度正常。
- `mlx-whisper` 的进度主要来自底层 tqdm frames，不一定每 5 秒输出标准 `[进度]` 行；看到 frames 百分比持续变化就是正常运行。
- FunASR 已作为备用引擎接入，默认不抢占 mlx-whisper；中文素材可用 `--engine funasr` 对比效果，或用 `--engine auto` 做失败回退。
- Apple Silicon 上 FunASR 默认走 MPS。25分09秒 B站音频实测：FunASR CPU 墙钟约 130.33 秒，FunASR MPS 墙钟约 96.82 秒；以后除非 MPS 报错，否则优先使用 `--funasr-device auto` 或 `--funasr-device mps`。
- 对长视频先报告音频时长和当前百分比，用户要求频繁汇报时，可以轮询运行中的 session，不要重启转写。
- `--transcribe-only` 原来只适合音频；现在传本地视频也会自动先提取音频再转写。

---

## 注意事项

1. **FFmpeg 路径**：如果 ffmpeg 不在系统 PATH 中，脚本会自动检测当前工作目录下的 `ffmpeg-master-latest-win64-gpl/bin/`
2. **CUDA vs CPU**：脚本自动检测 CUDA 可用性，不可用时回退到 CPU（速度慢10-20倍）
3. **GTX 1650 优化**：使用 int8 量化 + FP32 模式（`--fp16 False`），比默认 FP16 更快
4. **Windows 编码**：脚本避免使用 emoji，使用 `[OK]` `[错误]` 等文本标记
5. **网络问题**：B站下载可能需要代理或 Cookie，支持 `--cookie` 参数
6. **长时间运行**：批量处理多个视频时，建议告知用户预计耗时（约3-5分钟/视频 for 30分钟视频+GPU）
7. **转写耗时**：28分钟视频约3分钟（GPU），属于正常；脚本每5秒输出进度，不要因为长时间运行就中断命令
8. **输出缓冲**：脚本已设置 `sys.stdout.reconfigure(line_buffering=True)` 和 `flush=True`，确保进度信息实时输出
9. **音频格式**：下载或提取音频时优先保留原始 m4a/aac/opus，不要主动转 mp3（重编码慢、可能截断长视频）
10. **长视频命令超时**：WorkBuddy 的命令执行有超时机制，长视频（>15分钟）转写可能超时被 kill。建议长视频在终端手动运行
11. **本地 mp4 提取音频**：先用 `-c:a copy`，失败后再转 AAC；不要一上来 `-b:a 192k` 重编码

---

## 踩坑经验

- torch CUDA版安装：`pip install torch` 默认装 CPU 版，必须从 `https://download.pytorch.org/whl/cu121` 安装 CUDA 版，或手动下载 whl 文件本地安装
- Whisper FP16 vs FP32：GTX 系列（1650等老架构）FP16 性能极弱，必须加 `--fp16 False` 或用 Faster Whisper int8 量化
- yt-dlp B站下载：需要 ffmpeg 支持音频格式转换，否则只能下载原始格式
- Windows 终端编码：避免在 print 中使用 emoji，会导致 UnicodeEncodeError
- faster-whisper 模型下载：首次使用需从 HuggingFace 下载模型，网络不通时需配置代理或手动下载
- macOS Apple Silicon：faster-whisper (ctranslate2) 不支持 Apple GPU，必须用 mlx-whisper 代替，否则 CPU 跑极慢
- mlx-whisper 模型仓库：`mlx-community/whisper-small-mlx`（不是 `whisper-small`，会401报错）
- ffprobe 缺失：ffmpeg 便携版可能没有 ffprobe，需单独下载或用 ffmpeg -i 替代获取时长
- 本地视频路径带中文或空格时，必须整体加引号；脚本内部使用 subprocess 参数列表处理 ffmpeg，避免 shell 转义问题
