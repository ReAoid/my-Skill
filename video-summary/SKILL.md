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
2. Conda 环境检测 → 环境自检汇总 → Cookie 自动加载 → 硬件自动适配
  ↓
3. 网络视频先检查字幕：yt-dlp --list-subs
   ├── 有字幕 → 直接提取（秒级完成）
   └── 无字幕 → 下载原始音频 → ASR 转写（实时进度报告）
   本地视频 → ffmpeg 无损提取音轨 → ASR 转写
  ↓
4. 输出结构化 TXT（元信息 + 完整文本 + 带时间戳分段）
  ↓
5. 任务结束全局统计汇总（成功/失败/耗时）
```

---

## 环境规范

**本工具仅支持 Anaconda/Miniconda 虚拟环境运行**，原生 python venv 不再提供支持。
脚本启动时自动校验 conda 环境，非 conda 环境会输出红色警告并引导安装。

### 虚拟环境（固定名称 `bilibili_trans`）

```bash
# 创建环境，指定 Python 3.10
conda create -n bilibili_trans python=3.10 -y

# 激活环境
conda activate bilibili_trans

# 安装基础依赖
conda install ffmpeg -y
pip install yt-dlp opencc-python-reimplemented
```

### 按系统安装 GPU 依赖

#### Windows NVIDIA CUDA 显卡
```powershell
pip install torch==2.3.1+cu121 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper funasr modelscope soundfile
```

> GTX16xx 老显卡：脚本运行时**自动**强制关闭 FP16，使用 int8 量化提速

#### Windows 纯 CPU
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper funasr modelscope soundfile
```

#### macOS Apple Silicon (M系列)
```bash
pip install mlx-whisper torch torchvision torchaudio funasr modelscope soundfile
```

#### macOS Intel（仅 CPU 慢速运行）
```bash
pip install faster-whisper torch torchvision torchaudio funasr modelscope soundfile
```

---

## 依赖项

| 依赖 | 必需 | 说明 |
|------|------|------|
| yt-dlp | 是 | B站视频/字幕下载 |
| mlx-whisper | Apple Silicon | Metal GPU 加速转写 |
| faster-whisper | Windows/备选 | CUDA/CPU 转写 |
| ffmpeg | 是 | 音频提取（conda install ffmpeg） |
| opencc-python-reimplemented | 推荐 | 繁简转换 |
| torch | 推荐 | GPU 加速支持 |
| funasr / modelscope / soundfile | 备用 | 中文备用引擎 |

脚本启动时会自动输出**全量环境自检汇总面板**，包含：
- 系统信息与 Conda 环境
- yt-dlp / ffmpeg 状态
- CUDA / MPS 加速状态
- Whisper 引擎（mlx-whisper / faster-whisper）可用性
- FunASR 可用性

依赖缺失时，脚本会自动识别操作系统并输出对应的**一键安装命令**（区分 CUDA/CPU/MPS 版 torch）。

---

## Cookie 配置（B站防 403 限制）

脚本启动时**自动检测** `./cookie.txt`，存在则自动注入 yt-dlp，无需手动传参 `--cookie`。

```bash
# 将 cookie.txt 放在脚本同级目录，启动自动加载
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL"
```

如果文件缺失，脚本会自动打印 Cookie 导出配置步骤：
1. 浏览器安装插件：**Get Cookies LOCAL**
2. 打开 B 站主页，导出 cookies 为 txt 格式
3. 重命名为 `cookie.txt`，放置到脚本目录

---

## 硬件自动适配

脚本启动时自动检测硬件并**自动优化运行参数**：

| 硬件 | 自动优化 |
|------|---------|
| Apple Silicon (M1/M2/M3/M4) | 锁定 mlx-whisper (Metal GPU)，禁用 faster-whisper |
| Windows GTX 16xx / GTX 10xx | 强制关闭 FP16，int8 量化 |
| Windows 新架构 GPU | 提示可使用 medium/large 大模型 |
| 纯 CPU | 提示使用 tiny/base 轻量模型 |
| 所有平台 | 默认开启 `--cleanup-audio`，默认模型固定 `small` |

脚本还会输出详细的**硬件优化建议**，帮助用户了解当前配置的最佳实践。

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
- 视频超过20个时，**脚本自动弹出交互确认**，告知预计耗时，用户输入 y/n
- 支持用户指定范围，如 `--limit 10`

### Step 1b：Conda 环境检测（自动执行）

脚本启动时自动：
1. 检测系统是否存在 `conda` 命令，未检测到则输出 Miniconda 完整安装步骤
2. 读取 `CONDA_DEFAULT_ENV`，非 conda 环境输出警告并引导
3. 检测到 conda 但未创建环境时，自动输出全套创建/激活/安装指令

### Step 1c：Cookie 自动加载（自动执行）

脚本启动时自动检测 `./cookie.txt` 和脚本同级 `cookie.txt`，存在即加载。

### Step 1d：硬件检测与自动适配（自动执行）

脚本启动时自动检测硬件，适配最优参数，无需外部传参。

### Step 2：字幕优先策略

对每个视频先检查是否有可用字幕：

```bash
yt-dlp --list-subs "视频URL"
```

- 如果有 `zh-CN` / `zh-Hans` / `zh` 字幕 → 直接提取
- 如果只有弹幕（danmaku）或无字幕 → 走下载+转写路线

### Step 3：下载或提取音频（无字幕/本地视频时）

网络视频保留原始音频格式，不主动转 mp3。

### Step 4：ASR 转写（带双层降级容错）

脚本内置**双层降级容错机制**：

**第一层 - 模型大小降级**：
- large/medium 模型下载/加载失败 → 自动降级 small → tiny
- 确保即使在资源受限环境下也能完成转写

**第二层 - 引擎切换降级**：
- Whisper (mlx-whisper/faster-whisper) 全部失败 → 自动回退 FunASR
- FunASR 加载失败 → 自动切回 Whisper
- 提供最大容错保障

实时进度报告（每5秒输出百分比、已处理时间、片段数、预计剩余时间）。

### Step 4b：长视频超时风险提示

单视频时长 > 900 秒（15分钟），脚本**自动打印超时风险提示**，建议在本地终端手动运行避免进程被杀：

```
  ╔══════════════════════════════════════════════════════════╗
  ║  [长视频警告]                                          ║
  ║  当前视频时长: 73分24秒 (4404.0s)                      ║
  ║  超过15分钟阈值，转写可能需要较长时间                   ║
  ║                                                         ║
  ║  建议: 在本地终端手动运行，避免进程超时被 kill          ║
  ╚══════════════════════════════════════════════════════════╝
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

### Step 6：任务结束全局统计汇总

批量任务处理完成后，脚本自动输出统计汇总：

```
============================================================
  [任务统计] 执行完成汇总
============================================================
  总视频数:     12
  成功:         11
  失败:         1
  失败列表:     BV1xx411c7mD
  总运行耗时:   35分28秒
  输出目录:     /path/to/bilibili_output
============================================================
```

---

## 输出目录（自适应规则）

| 输入类型 | 默认输出目录 |
|---------|------------|
| B站单视频 | `./bilibili_output/BVxxx/` |
| UP主批量 | `./bilibili_output/BVxxx/`（统一在 bilibili_output 下） |
| 本地文件 | `原文件名_transcribe/`（与文件同目录） |
| `--output-dir` 自定义 | 按用户指定路径 |

```
bilibili_output/                     # B站视频输出
├── BV1xxxxxx/
│   ├── 视频标题.txt
│   └── 视频标题.m4a
├── progress.json
└── ...

本地视频.mp4 → 本地视频_transcribe/  # 本地文件输出
                └── 本地视频.txt
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

## 脚本调用方式

脚本已实现**完全独立运行**，无需 Agent 拼接命令，所有参数均有安全默认值：

```bash
# 最简单的用法（自动检测一切）
python -u <skill_dir>/scripts/bilibili_pipeline.py "https://www.bilibili.com/video/BV1xxxxxx/"

# 批量处理UP主视频
python -u <skill_dir>/scripts/bilibili_pipeline.py "https://space.bilibili.com/123456" --limit 10

# 处理本地文件（自动识别类型）
python -u <skill_dir>/scripts/bilibili_pipeline.py "本地视频.mp4"
python -u <skill_dir>/scripts/bilibili_pipeline.py "本地音频.m4a"

# 指定输出目录
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --output-dir "./my_output"

# 指定模型
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --model base

# 使用 FunASR 中文备用识别
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine funasr

# Whisper 失败时自动回退 FunASR
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --engine auto

# 不保留音频文件（默认开启）
python -u <skill_dir>/scripts/bilibili_pipeline.py "视频URL" --no-cleanup-audio

# 仅转写已有音频
python -u <skill_dir>/scripts/bilibili_pipeline.py --transcribe-only "音频路径"

# 仅从本地视频提取音频，不转写
python -u <skill_dir>/scripts/bilibili_pipeline.py --extract-audio "本地视频.mp4"
```

**重要**：不要使用 `2>&1` 重定向，直接运行即可。`-u` 参数禁用 Python 缓冲。

---

## 本地视频转写实战经验

这次处理 73分24秒本地 mp4 的经验：

- 源视频音轨本身是 AAC，最优方式是 `ffmpeg -vn -c:a copy` 直接抽成 `.m4a`，实际约 38 秒完成；误用 AAC 重编码会慢很多。
- 不要为了"通用"先转 mp3。mp3 转码多一步损耗，长视频还有截断和耗时风险；Whisper 可以直接吃 `.m4a`。
- Apple Silicon 上优先用 `mlx-whisper`，`small` 模型转写 73 分钟音频约 11 分钟，输出 2577 个片段，速度正常。
- `mlx-whisper` 的进度主要来自底层 tqdm frames，不一定每 5 秒输出标准 `[进度]` 行；看到 frames 百分比持续变化就是正常运行。
- FunASR 已作为备用引擎接入，默认不抢占 mlx-whisper；中文素材可用 `--engine funasr` 对比效果，或用 `--engine auto` 做失败回退。
- Apple Silicon 上 FunASR 默认走 MPS。25分09秒 B站音频实测：FunASR CPU 墙钟约 130.33 秒，FunASR MPS 墙钟约 96.82 秒；以后除非 MPS 报错，否则优先使用 `--funasr-device auto` 或 `--funasr-device mps`。
- 对长视频先报告音频时长和当前百分比，用户要求频繁汇报时，可以轮询运行中的 session，不要重启转写。
- `--transcribe-only` 原来只适合音频；现在传本地视频也会自动先提取音频再转写。

---

## 注意事项

1. **Conda 强制要求**：脚本启动强制检测 Conda 虚拟环境，必须激活 `bilibili_trans` 环境才能运行
2. **Cookie 自动加载**：`cookie.txt` 放在脚本目录自动生效，无需手动传入 `--cookie`
3. **硬件自动适配**：脚本自动检测硬件，自动优化 FP16/量化/模型大小，无需外部传参
4. **长视频保护**：>15分钟视频自动打印超时风险提示
5. **批量确认**：>20个视频自动弹出交互确认，询问用户是否继续
6. **模型降级容错**：双层降级保障（模型大小 + 引擎切换）
7. **环境自检汇总**：启动时打印完整系统状态面板
8. **依赖安装指引**：缺失依赖自动输出系统对应的一键安装命令
9. **FFmpeg 路径**：如果 ffmpeg 不在系统 PATH 中，脚本会自动检测当前工作目录下的 `ffmpeg-master-latest-win64-gpl/bin/`
10. **CUDA vs CPU**：脚本自动检测 CUDA 可用性，不可用时回退到 CPU（速度慢10-20倍）
11. **GTX 1650 优化**：脚本自动检测并优化（int8 量化 + FP32）
12. **Windows 编码**：脚本避免使用 emoji，使用 `[OK]` `[错误]` 等文本标记
13. **长时间运行**：批量处理多个视频时，脚本自动输出预计耗时
14. **转写耗时**：28分钟视频约3分钟（GPU），属于正常；脚本每5秒输出进度
15. **输出缓冲**：脚本已设置 `sys.stdout.reconfigure(line_buffering=True)` 和 `flush=True`
16. **音频格式**：下载或提取音频时优先保留原始 m4a/aac/opus，不要主动转 mp3
17. **长视频命令超时**：>15分钟视频脚本会提示在本地终端手动运行
18. **本地 mp4 提取音频**：先用 `-c:a copy`，失败后再转 AAC

---

## 踩坑经验

- torch CUDA版安装：`pip install torch` 默认装 CPU 版，必须从 `https://download.pytorch.org/whl/cu121` 安装 CUDA 版
- Whisper FP16 vs FP32：GTX 系列（1650等老架构）FP16 性能极弱，脚本自动强制关闭
- yt-dlp B站下载：需要 ffmpeg 支持音频格式转换，否则只能下载原始格式
- Windows 终端编码：避免在 print 中使用 emoji，会导致 UnicodeEncodeError
- faster-whisper 模型下载：首次使用需从 HuggingFace 下载模型
- macOS Apple Silicon：faster-whisper (ctranslate2) 不支持 Apple GPU，脚本自动使用 mlx-whisper
- mlx-whisper 模型仓库：`mlx-community/whisper-small-mlx`（不是 `whisper-small`，会401报错）
- ffprobe 缺失：ffmpeg 便携版可能没有 ffprobe，脚本自动使用 `ffmpeg -i` 替代
- Conda 环境创建后需重新激活：`conda activate bilibili_trans` 后再安装依赖
- 本地视频路径带中文或空格时，必须整体加引号
