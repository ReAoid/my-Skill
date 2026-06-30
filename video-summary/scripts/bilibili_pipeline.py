#!/usr/bin/env python3
"""
B站视频批量转写 Pipeline
功能：字幕优先提取 → 无字幕时下载音频 → Whisper 转写 → 生成 TXT
支持：单个视频、UP主批量、断点续传
macOS：自动检测 Apple Silicon，优先用 mlx-whisper (Metal GPU)
Windows：使用 faster-whisper (CUDA/CPU)
"""

import os
import sys
import json
import subprocess
import re
import argparse
import shutil
import gc
import platform
from pathlib import Path
from datetime import datetime

# 繁简转换：Whisper 经常输出繁体，统一转简体
try:
    from opencc import OpenCC
    _t2s_converter = OpenCC('t2s')  # 繁体 → 简体
    def to_simplified(text):
        """将繁体中文转换为简体中文"""
        return _t2s_converter.convert(text)
except ImportError:
    def to_simplified(text):
        """opencc 未安装，不转换"""
        return text

# 强制 stdout/stderr 无缓冲
sys.stdout.reconfigure(line_buffering=True)
sys.stderr = sys.stdout
os.environ["PYTHONUNBUFFERED"] = "1"

# macOS 清除代理，避免 HuggingFace 503
if platform.system() == "Darwin":
    for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
        os.environ.pop(k, None)


# ============ 配置 ============
DEFAULT_MODEL = "small"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_ENGINE = "whisper"
DEFAULT_FUNASR_MODEL = "paraformer-zh"
DEFAULT_FUNASR_DEVICE = "auto"
DEFAULT_OUTPUT_DIR = "./bilibili_output"
MODEL_RELEASE_INTERVAL = 2  # 每处理 N 个视频释放一次模型
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".opus", ".ogg", ".webm"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".wmv", ".m4v", ".webm"}

# macOS mlx-whisper 模型仓库映射
MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
}

# 全局模型缓存（用于跨视频复用，定期释放）
_cached_model = None
_cached_model_key = None


def is_apple_silicon():
    """检测是否为 Apple Silicon (M1/M2/M3/M4)"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def run_command(cmd, cwd=None, env=None, timeout=None):
    """运行 shell 命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True,
            cwd=cwd, env=env, timeout=timeout
        )
        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                stdout = result.stdout.decode(enc)
                stderr = result.stderr.decode(enc)
                return result.returncode == 0, stdout, stderr
            except (UnicodeDecodeError, AttributeError):
                continue
        return result.returncode == 0, "", ""
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"


def safe_filename(name, fallback="media"):
    """生成可跨平台使用的文件名。"""
    safe = "".join(
        c for c in name
        if c.isalnum() or c in (' ', '-', '_', '.') or '\u4e00' <= c <= '\u9fff'
    ).strip()
    return safe or fallback


def is_audio_file(path):
    return Path(path).suffix.lower() in AUDIO_EXTS


def is_video_file(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def find_ffmpeg_dir():
    """在系统中搜索 ffmpeg 所在目录"""
    which_result = shutil.which("ffmpeg")
    if which_result:
        return Path(which_result).parent

    candidates = [
        Path.cwd() / "ffmpeg-master-latest-win64-gpl" / "bin",
        Path(__file__).resolve().parent / "ffmpeg-master-latest-win64-gpl" / "bin",
        Path(__file__).resolve().parent.parent / "ffmpeg-master-latest-win64-gpl" / "bin",
        Path.home() / "Desktop" / "ffmpeg-master-latest-win64-gpl" / "bin",
        Path.home() / "Downloads" / "ffmpeg-master-latest-win64-gpl" / "bin",
    ]
    try:
        p = Path.cwd()
        for _ in range(5):
            candidates.insert(0, p / "ffmpeg-master-latest-win64-gpl" / "bin")
            p = p.parent
    except Exception:
        pass

    for p in candidates:
        try:
            # 检查 ffmpeg.exe (Windows) 或 ffmpeg (macOS/Linux)
            if p.exists() and ((p / "ffmpeg.exe").exists() or (p / "ffmpeg").exists()):
                return p
        except Exception:
            continue

    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            if d and (Path(d, "ffmpeg.exe").exists() or Path(d, "ffmpeg").exists()):
                return Path(d)
        except Exception:
            continue

    return None


def get_ffmpeg_env():
    """获取包含 FFmpeg 的环境变量"""
    env = os.environ.copy()
    # macOS 清除代理
    if platform.system() == "Darwin":
        for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
            env.pop(k, None)
    ff_dir = find_ffmpeg_dir()
    if ff_dir:
        ff_str = str(ff_dir)
        if ff_str not in env.get("PATH", ""):
            env["PATH"] = ff_str + os.pathsep + env.get("PATH", "")
    return env


def check_subtitles(video_url, env=None):
    """检查视频是否有可用字幕，返回字幕语言列表"""
    cmd = f'yt-dlp --list-subs "{video_url}"'
    success, stdout, stderr = run_command(cmd, env=env, timeout=30)
    if not success or not stdout:
        return []

    subs = []
    in_subs_section = False
    for line in stdout.split('\n'):
        if 'Available subtitles' in line:
            in_subs_section = True
            continue
        if in_subs_section:
            if line.strip() == '' or 'Available automatic captions' in line:
                if subs:
                    break
                continue
            parts = line.strip().split()
            if parts:
                lang = parts[0]
                if lang.startswith('zh'):
                    subs.append(lang)

    in_auto_section = False
    for line in stdout.split('\n'):
        if 'Available automatic captions' in line:
            in_auto_section = True
            continue
        if in_auto_section and line.strip():
            parts = line.strip().split()
            if parts:
                lang = parts[0]
                if lang.startswith('zh') and lang not in subs:
                    subs.append(lang)
    return subs


def extract_subtitles(video_url, output_dir, env=None):
    """提取视频字幕，返回字幕文件路径"""
    for lang in ['zh-CN', 'zh-Hans', 'zh']:
        cmd = f'yt-dlp --write-subs --sub-langs {lang} --skip-download ' \
              f'-o "{output_dir}/%(title)s.%(ext)s" "{video_url}"'
        success, stdout, stderr = run_command(cmd, env=env, timeout=60)
        if success:
            for ext in ['.srt', '.vtt', '.json', '.ass']:
                files = list(output_dir.glob(f"*{ext}"))
                if files:
                    return files[0]
    return None


def parse_srt(srt_path):
    """解析 SRT 字幕文件，返回带时间戳的文本列表"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n(.*?)(?=\n\d+\n|\Z)',
        re.DOTALL
    )

    for match in pattern.finditer(content):
        start_str = match.group(1).replace(',', '.')
        end_str = match.group(2).replace(',', '.')
        text = match.group(3).strip().replace('\n', ' ')

        if not text or text.startswith('['):
            continue

        def parse_time(t):
            parts = t.split(':')
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s

        segments.append({
            'start': parse_time(start_str),
            'end': parse_time(end_str),
            'text': text
        })

    return segments


def parse_vtt(vtt_path):
    """解析 VTT 字幕文件"""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    segments = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*\n(.*?)(?=\n\n|\Z)',
        re.DOTALL
    )

    for match in pattern.finditer(content):
        start_str = match.group(1).replace(',', '.')
        end_str = match.group(2).replace(',', '.')
        text = match.group(3).strip().replace('\n', ' ')

        if not text or text.startswith('['):
            continue

        def parse_time(t):
            parts = t.split(':')
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s

        segments.append({
            'start': parse_time(start_str),
            'end': parse_time(end_str),
            'text': text
        })

    return segments


def extract_video_info(video_url, env=None):
    """提取视频信息"""
    cmd = f'yt-dlp --dump-json --no-download "{video_url}"'
    success, stdout, stderr = run_command(cmd, env=env, timeout=30)

    if not success or not stdout:
        print(f"[错误] 获取信息失败: {stderr}")
        return None

    try:
        info = json.loads(stdout)
        return {
            "id": info.get("id"),
            "title": info.get("title", "unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "unknown"),
            "upload_date": info.get("upload_date", ""),
            "description": info.get("description", ""),
        }
    except json.JSONDecodeError:
        print("[错误] 解析视频信息失败")
        return None


def download_audio(video_url, output_dir, env=None):
    """下载 B站视频音频（保留原始格式 m4a，避免 mp3 转换截断长视频）"""
    print(f"[下载] 正在下载音频...", flush=True)

    output_template = str(output_dir / "%(title)s.%(ext)s")
    cmd = f'yt-dlp -x --audio-quality 0 --quiet --no-warnings ' \
          f'-o "{output_template}" "{video_url}"'

    success, stdout, stderr = run_command(cmd, env=env, timeout=600)

    if not success:
        print(f"[错误] 下载失败: {stderr}", flush=True)
        return None

    audio_files = []
    for ext in ['*.m4a', '*.opus', '*.webm', '*.mp3', '*.wav', '*.ogg']:
        audio_files.extend(output_dir.glob(ext))
    if audio_files:
        return sorted(audio_files, key=lambda f: f.stat().st_mtime)[-1]
    return None


def extract_audio_from_local_video(video_path, output_dir, env=None):
    """从本地视频提取音频，优先无损复制音轨，失败后再转 AAC。"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{safe_filename(video_path.stem, 'local_video')}.m4a"

    print("[提取] 从本地视频提取音频...", flush=True)
    print("[提取] 优先使用 -c:a copy 直接封装音轨，避免重编码耗时和截断风险", flush=True)
    copy_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-c:a", "copy", str(audio_path)
    ]
    result = subprocess.run(copy_cmd, capture_output=True, env=env)
    if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
        print(f"[完成] 音频已提取: {audio_path}", flush=True)
        return audio_path

    print("[警告] 直接复制音轨失败，改用 AAC 转码兜底", flush=True)
    transcode_cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-c:a", "aac", "-b:a", "128k", str(audio_path)
    ]
    result = subprocess.run(transcode_cmd, capture_output=True, env=env)
    if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
        print(f"[完成] 音频已提取: {audio_path}", flush=True)
        return audio_path

    stderr = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
    print(f"[错误] 本地视频音频提取失败: {stderr}", flush=True)
    return None


def _get_audio_duration(audio_path):
    """获取音频文件时长（秒），兼容无 ffprobe 的情况"""
    # 先试 ffprobe
    cmd = f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{audio_path}"'
    success, stdout, stderr = run_command(cmd, timeout=10)
    if success and stdout.strip():
        try:
            return float(stdout.strip())
        except ValueError:
            pass

    # ffprobe 不可用时，用 ffmpeg -i 替代
    cmd = f'ffmpeg -i "{audio_path}" 2>&1'
    success, stdout, stderr = run_command(cmd, timeout=10)
    output = (stdout or '') + (stderr or '')
    for line in output.split('\n'):
        if 'Duration:' in line:
            m = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', line)
            if m:
                h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return h * 3600 + mi * 60 + s

    return None


def release_model():
    """释放缓存的模型，回收内存"""
    global _cached_model, _cached_model_key
    if _cached_model is not None:
        print("[内存] 释放识别模型，回收内存...", flush=True)
        del _cached_model
        _cached_model = None
        _cached_model_key = None
        gc.collect()
        print("[内存] 模型已释放", flush=True)


def transcribe_with_mlx_whisper(audio_path, model_size=DEFAULT_MODEL):
    """使用 mlx-whisper (Apple Silicon Metal GPU) 转写音频"""
    global _cached_model, _cached_model_key

    import mlx_whisper
    import time

    model_key = f"mlx-{model_size}"
    repo_id = MLX_MODEL_REPOS.get(model_size, MLX_MODEL_REPOS["small"])

    # 获取音频时长
    audio_duration = _get_audio_duration(audio_path)
    if audio_duration:
        dur_min = int(audio_duration // 60)
        dur_sec = int(audio_duration % 60)
        print(f"[信息] 音频时长: {dur_min}分{dur_sec}秒 ({audio_duration:.1f}s)", flush=True)

    print(f"[转写] mlx-whisper ({model_size}, Apple Silicon GPU)...", flush=True)
    start_time = time.time()

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo_id,
        language="zh",
        verbose=False,
    )

    elapsed = time.time() - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)

    full_text = []
    segments_with_time = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            full_text.append(text)
            segments_with_time.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            })

    print(f"[进度] 100.0% | 转写完成 | 用时 {elapsed_min}分{elapsed_sec}秒 | 共 {len(segments_with_time)} 个片段", flush=True)
    return full_text, segments_with_time


def transcribe_with_faster_whisper(audio_path, model_size=DEFAULT_MODEL,
                                    compute_type=DEFAULT_COMPUTE_TYPE):
    """使用 Faster Whisper 转写音频（Windows CUDA/CPU），实时报告进度"""
    global _cached_model, _cached_model_key

    from faster_whisper import WhisperModel
    import torch
    import time

    model_key = f"fw-{model_size}-{compute_type}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[转写] Faster Whisper ({model_size}, {compute_type}, {device})...", flush=True)

    # 获取音频总时长
    audio_duration = _get_audio_duration(audio_path)
    if audio_duration:
        dur_min = int(audio_duration // 60)
        dur_sec = int(audio_duration % 60)
        print(f"[信息] 音频时长: {dur_min}分{dur_sec}秒 ({audio_duration:.1f}s)", flush=True)

    # 加载或复用模型
    if _cached_model is not None and _cached_model_key == model_key:
        print(f"[转写] 复用已加载模型...", flush=True)
        model = _cached_model
    else:
        # 释放旧模型
        release_model()
        print(f"[转写] 正在加载模型 {model_size}...", flush=True)
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _cached_model = model
        _cached_model_key = model_key
        print(f"[转写] 模型加载完成，开始转写...", flush=True)

    segments, info = model.transcribe(
        str(audio_path),
        language="zh",
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
    )

    print(f"[信息] 检测到语言: {info.language}, 概率: {info.language_probability:.2f}", flush=True)

    full_text = []
    segments_with_time = []
    segment_count = 0
    last_progress_report = time.time()
    progress_interval = 5
    start_time = time.time()

    for segment in segments:
        text = segment.text.strip()
        if text:
            full_text.append(text)
            segments_with_time.append({
                "start": segment.start,
                "end": segment.end,
                "text": text
            })
            segment_count += 1

            now = time.time()
            if now - last_progress_report >= progress_interval:
                last_progress_report = now
                elapsed = now - start_time
                if audio_duration and segment.end > 0:
                    progress_pct = min(segment.end / audio_duration * 100, 99.9)
                    if segment.end > 0:
                        speed = segment.end / elapsed
                        remaining = (audio_duration - segment.end) / speed if speed > 0 else 0
                        rem_min = int(remaining // 60)
                        rem_sec = int(remaining % 60)
                        eta_str = f", 预计剩余 {rem_min}分{rem_sec}秒"
                    else:
                        eta_str = ""
                    print(f"[进度] {progress_pct:5.1f}% | 已处理 {segment.end:.0f}s / {audio_duration:.0f}s | 片段数: {segment_count}{eta_str}", flush=True)
                else:
                    elapsed_min = int(elapsed // 60)
                    elapsed_sec = int(elapsed % 60)
                    print(f"[进度] 已运行 {elapsed_min}分{elapsed_sec}秒 | 当前位置: {segment.end:.0f}s | 片段数: {segment_count}", flush=True)

    elapsed = time.time() - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    print(f"[进度] 100.0% | 转写完成 | 用时 {elapsed_min}分{elapsed_sec}秒 | 共 {segment_count} 个片段", flush=True)

    return full_text, segments_with_time


def _normalize_funasr_time(value, audio_duration=None):
    """FunASR 的时间字段常见为毫秒；如果看起来像毫秒就转为秒。"""
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value > 1000 or (audio_duration and value > audio_duration * 1.5):
        return value / 1000.0
    return value


def _clean_funasr_text(text):
    """清理 SenseVoice/FunASR 可能带出的控制标签。"""
    text = re.sub(r"<\|[^|]+?\|>", "", text or "")
    return text.strip()


def _approximate_segments_from_text(text, duration):
    """没有句级时间戳时，按标点切句并用字符占比估算时间。"""
    text = _clean_funasr_text(text)
    if not text:
        return [], []

    pieces = [
        p.strip()
        for p in re.split(r"(?<=[。！？!?；;])\s*", text)
        if p.strip()
    ]
    if not pieces:
        pieces = [text]

    duration = float(duration or 0)
    if duration <= 0:
        duration = max(len(text) / 4.0, 1.0)

    total_chars = max(sum(len(p) for p in pieces), 1)
    cursor = 0.0
    segments = []
    for idx, piece in enumerate(pieces):
        if idx == len(pieces) - 1:
            end = duration
        else:
            end = min(duration, cursor + duration * len(piece) / total_chars)
        segments.append({"start": cursor, "end": max(end, cursor), "text": piece})
        cursor = end
    return pieces, segments


def choose_funasr_device(device=DEFAULT_FUNASR_DEVICE):
    """选择 FunASR 推理设备；Apple Silicon 默认用 MPS。"""
    if device != "auto":
        return device

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if is_apple_silicon() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_with_funasr(audio_path, model_name=DEFAULT_FUNASR_MODEL,
                           device=DEFAULT_FUNASR_DEVICE):
    """使用 FunASR 中文模型转写音频，作为 Whisper 的备用识别引擎。"""
    global _cached_model, _cached_model_key

    import time
    from funasr import AutoModel

    audio_duration = _get_audio_duration(audio_path)
    if audio_duration:
        dur_min = int(audio_duration // 60)
        dur_sec = int(audio_duration % 60)
        print(f"[信息] 音频时长: {dur_min}分{dur_sec}秒 ({audio_duration:.1f}s)", flush=True)

    device = choose_funasr_device(device)
    model_key = f"funasr-{model_name}-{device}"
    print(f"[转写] FunASR ({model_name}, {device})...", flush=True)

    if _cached_model is not None and _cached_model_key == model_key:
        print("[转写] 复用已加载 FunASR 模型...", flush=True)
        model = _cached_model
    else:
        release_model()
        print(f"[转写] 正在加载 FunASR 模型 {model_name}...", flush=True)
        model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            punc_model="ct-punc-c",
            disable_update=True,
            disable_pbar=False,
            device=device,
        )
        _cached_model = model
        _cached_model_key = model_key
        print("[转写] FunASR 模型加载完成，开始转写...", flush=True)

    start_time = time.time()
    result = model.generate(
        input=str(audio_path),
        batch_size_s=300,
        merge_vad=True,
        merge_length_s=15,
        sentence_timestamp=True,
    )

    item = result[0] if isinstance(result, list) and result else {}
    sentence_info = item.get("sentence_info") or []
    segments_with_time = []

    for sent in sentence_info:
        text = _clean_funasr_text(sent.get("sentence") or sent.get("text") or "")
        if not text:
            continue
        segments_with_time.append({
            "start": _normalize_funasr_time(sent.get("start"), audio_duration),
            "end": _normalize_funasr_time(sent.get("end"), audio_duration),
            "text": text,
        })

    if segments_with_time:
        full_text = [seg["text"] for seg in segments_with_time]
    else:
        text = _clean_funasr_text(item.get("text", ""))
        full_text, segments_with_time = _approximate_segments_from_text(text, audio_duration)
        if segments_with_time:
            print("[提示] FunASR 未返回句级时间戳，已按文本长度估算时间轴", flush=True)

    elapsed = time.time() - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)
    print(f"[进度] 100.0% | FunASR 转写完成 | 用时 {elapsed_min}分{elapsed_sec}秒 | 共 {len(segments_with_time)} 个片段", flush=True)
    return full_text, segments_with_time


def transcribe_audio(audio_path, model_size=DEFAULT_MODEL,
                     compute_type=DEFAULT_COMPUTE_TYPE, engine=DEFAULT_ENGINE,
                     funasr_model=DEFAULT_FUNASR_MODEL,
                     funasr_device=DEFAULT_FUNASR_DEVICE):
    """按当前平台选择 Whisper 后端，返回转写数据和方法说明。"""
    if engine == "funasr":
        device = choose_funasr_device(funasr_device)
        transcript_data = transcribe_with_funasr(
            audio_path, model_name=funasr_model, device=device
        )
        method = f"FunASR ({funasr_model}, {device})"
    elif engine == "auto":
        try:
            transcript_data, method = transcribe_audio(
                audio_path,
                model_size=model_size,
                compute_type=compute_type,
                engine="whisper",
                funasr_model=funasr_model,
                funasr_device=funasr_device,
            )
        except Exception as e:
            print(f"[警告] Whisper 转写失败，改用 FunASR 备用: {e}", flush=True)
            device = choose_funasr_device(funasr_device)
            transcript_data = transcribe_with_funasr(
                audio_path, model_name=funasr_model, device=device
            )
            method = f"FunASR fallback ({funasr_model}, {device})"
    elif is_apple_silicon():
        transcript_data = transcribe_with_mlx_whisper(audio_path, model_size=model_size)
        method = f"mlx-whisper ({model_size}, Apple Silicon GPU)"
    else:
        transcript_data = transcribe_with_faster_whisper(
            audio_path, model_size=model_size, compute_type=compute_type
        )
        method = f"Faster Whisper ({model_size}, {compute_type})"
    return transcript_data, method


def save_transcript_txt(video_info, transcript_data, output_dir, method="Whisper"):
    """将转写结果保存为结构化 TXT 文件"""
    title = video_info.get("title", "unknown")
    video_url = video_info.get("url", "")
    uploader = video_info.get("uploader", "unknown")
    upload_date = video_info.get("upload_date", "")
    duration = video_info.get("duration", 0)
    video_id = video_info.get("id", "unknown")

    full_text, segments_with_time = transcript_data

    # 繁体 → 简体转换
    full_text = [to_simplified(t) for t in full_text]
    segments_with_time = [
        {"start": s["start"], "end": s["end"], "text": to_simplified(s["text"])}
        for s in segments_with_time
    ]
    title = to_simplified(title)

    if duration:
        duration_int = int(duration)
        duration_str = f"{duration_int // 60}分{duration_int % 60}秒"
    else:
        duration_str = "未知"

    if upload_date and len(upload_date) == 8:
        upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

    lines = []
    lines.append(f"TITLE: {title}")
    lines.append(f"ID: {video_id}")
    lines.append(f"URL: {video_url}")
    lines.append(f"UPLOADER: {uploader}")
    lines.append(f"DATE: {upload_date}")
    lines.append(f"DURATION: {duration_str}")
    lines.append(f"METHOD: {method}")
    lines.append(f"PROCESSED_AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"SEGMENT_COUNT: {len(segments_with_time)}")
    lines.append("")
    lines.append("=== FULL TEXT ===")
    lines.append(''.join(full_text))
    lines.append("")
    lines.append("=== TIMESTAMPS ===")

    for seg in segments_with_time:
        start_min = int(seg["start"] // 60)
        start_sec = int(seg["start"] % 60)
        end_min = int(seg["end"] // 60)
        end_sec = int(seg["end"] % 60)
        time_range = f"[{start_min:02d}:{start_sec:02d} -> {end_min:02d}:{end_sec:02d}]"
        lines.append(f"{time_range} {seg['text']}")

    lines.append("")
    lines.append("=== END ===")

    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_', '.') or '\u4e00' <= c <= '\u9fff').rstrip()
    if not safe_title:
        safe_title = video_id
    txt_path = output_dir / f"{safe_title}.txt"

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return txt_path


def load_progress(output_dir):
    """加载断点续传进度"""
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed": [], "failed": []}


def save_progress(output_dir, progress):
    """保存断点续传进度"""
    progress_file = output_dir / "progress.json"
    progress["last_update"] = datetime.now().isoformat()
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def process_local_media(media_path, output_dir=None, model_size=DEFAULT_MODEL,
                        compute_type=DEFAULT_COMPUTE_TYPE, cleanup_audio=False,
                        engine=DEFAULT_ENGINE, funasr_model=DEFAULT_FUNASR_MODEL,
                        funasr_device=DEFAULT_FUNASR_DEVICE):
    """处理本地音频或视频文件：视频先提取音频，再转写为 TXT。"""
    env = get_ffmpeg_env()
    media_path = Path(media_path)
    if not media_path.exists():
        print(f"[错误] 文件不存在: {media_path}")
        return False

    base_output_dir = Path(output_dir) if output_dir else media_path.parent / f"{safe_filename(media_path.stem)}_transcribe"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"开始处理本地文件: {media_path}")
    print(f"{'='*70}")

    audio_path = media_path
    if is_video_file(media_path):
        audio_path = extract_audio_from_local_video(media_path, base_output_dir, env=env)
        if not audio_path:
            return False
    elif not is_audio_file(media_path):
        print(f"[错误] 不支持的本地文件类型: {media_path.suffix}")
        return False

    try:
        transcript_data, method = transcribe_audio(
            audio_path,
            model_size=model_size,
            compute_type=compute_type,
            engine=engine,
            funasr_model=funasr_model,
            funasr_device=funasr_device,
        )
    except Exception as e:
        print(f"[错误] 转写失败: {e}")
        return False

    duration = _get_audio_duration(audio_path) or 0
    video_info = {
        "title": media_path.stem,
        "url": str(media_path),
        "uploader": "local",
        "upload_date": "",
        "duration": duration,
        "id": media_path.stem,
    }
    txt_path = save_transcript_txt(video_info, transcript_data, base_output_dir, method=method)
    print(f"[完成] TXT 已保存: {txt_path}", flush=True)

    if cleanup_audio and audio_path != media_path:
        audio_path.unlink()
        print("[清理] 已删除提取出的音频文件", flush=True)

    return True


def process_video(video_url, output_dir, model_size=DEFAULT_MODEL,
                  compute_type=DEFAULT_COMPUTE_TYPE, cleanup_audio=False,
                  video_index=0, engine=DEFAULT_ENGINE,
                  funasr_model=DEFAULT_FUNASR_MODEL,
                  funasr_device=DEFAULT_FUNASR_DEVICE):
    """处理单个视频：字幕优先 → 下载转写 → 输出 TXT"""
    env = get_ffmpeg_env()

    print(f"\n{'='*70}")
    print(f"开始处理: {video_url}")
    print(f"{'='*70}")

    # 1. 获取视频信息
    video_info = extract_video_info(video_url, env=env)
    if not video_info:
        print("[错误] 无法获取视频信息，跳过")
        return False

    video_info["url"] = video_url
    print(f"[标题] {video_info['title']}")
    print(f"[UP主] {video_info['uploader']}")
    dur = int(video_info['duration'])
    print(f"[时长] {dur // 60}分{dur % 60}秒")

    # 检查是否已处理
    progress = load_progress(output_dir)
    if video_info["id"] in progress["processed"]:
        print(f"[跳过] 已处理过: {video_info['id']}")
        return True

    # 创建视频输出目录
    video_output_dir = output_dir / video_info["id"]
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 字幕优先策略
    transcript_data = None
    method = ""

    print("[检查] 检查可用字幕...")
    sub_langs = check_subtitles(video_url, env=env)

    if sub_langs:
        print(f"[字幕] 发现字幕: {sub_langs}")
        sub_path = extract_subtitles(video_url, video_output_dir, env=env)
        if sub_path:
            print(f"[字幕] 字幕已提取: {sub_path.name}")
            if sub_path.suffix == '.srt':
                segments = parse_srt(sub_path)
            elif sub_path.suffix == '.vtt':
                segments = parse_vtt(sub_path)
            else:
                segments = parse_srt(sub_path)

            if segments:
                full_text = [seg['text'] for seg in segments]
                transcript_data = (full_text, segments)
                method = f"字幕提取 ({sub_path.suffix})"
                print(f"[字幕] 解析到 {len(segments)} 个片段")

    # 3. 无字幕，走下载+转写
    if not transcript_data:
        print("[转写] 无可用字幕，下载音频转写...")
        audio_path = download_audio(video_url, video_output_dir, env=env)
        if not audio_path:
            print("[错误] 音频下载失败，跳过")
            progress["failed"].append(video_info["id"])
            save_progress(output_dir, progress)
            return False
        print(f"[完成] 音频已下载: {audio_path.name}")

        try:
            transcript_data, method = transcribe_audio(
                audio_path,
                model_size=model_size,
                compute_type=compute_type,
                engine=engine,
                funasr_model=funasr_model,
                funasr_device=funasr_device,
            )
        except Exception as e:
            print(f"[错误] 转写失败: {e}")
            progress["failed"].append(video_info["id"])
            save_progress(output_dir, progress)
            return False

        # 清理音频
        if cleanup_audio:
            audio_path.unlink()
            print(f"[清理] 已删除音频文件")

        # 定期释放模型，回收内存
        if video_index > 0 and video_index % MODEL_RELEASE_INTERVAL == 0:
            release_model()

    # 4. 保存原始 TXT
    txt_path = save_transcript_txt(video_info, transcript_data, video_output_dir, method=method)
    print(f"[完成] TXT 已保存: {txt_path.name}")

    # 5. 更新进度
    progress["processed"].append(video_info["id"])
    save_progress(output_dir, progress)

    print(f"\n[成功] 处理完成！输出: {video_output_dir}")
    return True


def get_channel_videos(channel_url, limit=None, env=None):
    """获取 UP 主的所有视频链接"""
    print(f"[信息] 获取 UP 主视频列表: {channel_url}")

    cmd = f'yt-dlp --flat-playlist --print id "{channel_url}"'
    success, stdout, stderr = run_command(cmd, env=env, timeout=120)

    if not success:
        print(f"[错误] 获取视频列表失败: {stderr}")
        return []

    video_ids = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
    print(f"[信息] 找到 {len(video_ids)} 个视频")

    if limit and len(video_ids) > limit:
        print(f"[信息] 限制处理前 {limit} 个视频")
        video_ids = video_ids[:limit]

    video_urls = [f"https://www.bilibili.com/video/{vid}" for vid in video_ids]
    return video_urls


def main():
    global MODEL_RELEASE_INTERVAL
    parser = argparse.ArgumentParser(description="B站视频批量转写 Pipeline")
    parser.add_argument("url", nargs="?", help="视频URL、UP主空间URL、本地音频或本地视频路径")
    parser.add_argument("--limit", type=int, default=None, help="限制处理视频数量（默认不限制）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 模型大小")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE, help="量化类型")
    parser.add_argument("--engine", default=DEFAULT_ENGINE,
                        choices=["whisper", "funasr", "auto"],
                        help="转写引擎：whisper 默认；funasr 作为中文备用；auto 为 Whisper 失败后回退 FunASR")
    parser.add_argument("--funasr-model", default=DEFAULT_FUNASR_MODEL,
                        help="FunASR 模型名，默认 paraformer-zh")
    parser.add_argument("--funasr-device", default=DEFAULT_FUNASR_DEVICE,
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="FunASR 推理设备：auto 在 Apple Silicon 上优先用 mps")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--cleanup-audio", action="store_true", help="转写后删除音频文件")
    parser.add_argument("--transcribe-only", help="仅转写指定音频文件（跳过下载）")
    parser.add_argument("--extract-audio", help="仅从本地视频提取音频，不执行转写")
    parser.add_argument("--cookie", help="yt-dlp Cookie 文件路径")
    parser.add_argument("--release-interval", type=int, default=MODEL_RELEASE_INTERVAL,
                        help=f"每处理N个视频释放模型（默认{MODEL_RELEASE_INTERVAL}）")
    args = parser.parse_args()

    MODEL_RELEASE_INTERVAL = args.release_interval

    output_dir_was_set = args.output_dir != DEFAULT_OUTPUT_DIR
    output_dir = Path(args.output_dir)

    # 仅提取本地视频音频
    if args.extract_audio:
        video_path = Path(args.extract_audio)
        if not video_path.exists():
            print(f"[错误] 文件不存在: {video_path}")
            sys.exit(1)
        if not is_video_file(video_path):
            print(f"[错误] --extract-audio 需要本地视频文件: {video_path}")
            sys.exit(1)
        audio_path = extract_audio_from_local_video(video_path, output_dir, env=get_ffmpeg_env())
        if not audio_path:
            sys.exit(1)
        print(f"[完成] 已生成音频: {audio_path}")
        return

    # 仅转写模式
    if args.transcribe_only:
        audio_path = Path(args.transcribe_only)
        if not audio_path.exists():
            print(f"[错误] 文件不存在: {audio_path}")
            sys.exit(1)
        if is_video_file(audio_path):
            success = process_local_media(
                audio_path, output_dir=output_dir if output_dir_was_set else None, model_size=args.model,
                compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
                engine=args.engine, funasr_model=args.funasr_model,
                funasr_device=args.funasr_device
            )
            sys.exit(0 if success else 1)
        transcript_data, method = transcribe_audio(
            audio_path,
            model_size=args.model,
            compute_type=args.compute_type,
            engine=args.engine,
            funasr_model=args.funasr_model,
            funasr_device=args.funasr_device,
        )
        duration = _get_audio_duration(audio_path) or 0
        video_info = {
            "title": audio_path.stem,
            "url": "",
            "uploader": "unknown",
            "upload_date": "",
            "duration": duration,
            "id": audio_path.stem,
        }
        transcribe_output_dir = output_dir if output_dir_was_set else audio_path.parent
        transcribe_output_dir.mkdir(parents=True, exist_ok=True)
        txt_path = save_transcript_txt(video_info, transcript_data,
                                        transcribe_output_dir, method=method)
        print(f"[完成] 已生成: {txt_path}")
        return

    if not args.url:
        print("[用法] python bilibili_pipeline.py <视频URL或UP主主页URL> [选项]")
        print("  示例: python bilibili_pipeline.py https://www.bilibili.com/video/BV1xx411c7mD")
        print("        python bilibili_pipeline.py https://space.bilibili.com/123456 --limit 10")
        sys.exit(1)

    input_url = args.url

    # 本地音视频路径：自动选择本地处理流程
    local_path = Path(input_url).expanduser()
    if local_path.exists():
        success = process_local_media(
            local_path, output_dir=output_dir if output_dir_was_set else None, model_size=args.model,
            compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
            engine=args.engine, funasr_model=args.funasr_model,
            funasr_device=args.funasr_device
        )
        sys.exit(0 if success else 1)

    if args.cookie:
        os.environ["YT_DLP_COOKIE"] = args.cookie

    env = get_ffmpeg_env()

    # 检查依赖
    print("[检查] 检查依赖...")
    success, _, _ = run_command("yt-dlp --version")
    if not success:
        print("[错误] 未找到 yt-dlp: pip install yt-dlp")
        sys.exit(1)
    print("[OK] yt-dlp")

    if is_apple_silicon():
        try:
            import mlx_whisper
            print("[OK] mlx-whisper (Apple Silicon GPU)")
        except ImportError:
            print("[警告] 未找到 mlx-whisper，回退到 faster-whisper (CPU很慢)")
            print("  安装: pip install mlx-whisper")
    else:
        try:
            from faster_whisper import WhisperModel
            print("[OK] faster-whisper")
        except ImportError:
            print("[错误] 未找到 faster-whisper: pip install faster-whisper")
            sys.exit(1)

        import torch
        if torch.cuda.is_available():
            print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("[警告] CUDA 不可用，将使用 CPU（速度很慢）")

    if args.engine in ("funasr", "auto"):
        try:
            from funasr import AutoModel
            print("[OK] FunASR")
            print(f"[OK] FunASR device: {choose_funasr_device(args.funasr_device)}")
        except ImportError:
            if args.engine == "funasr":
                print("[错误] 未找到 FunASR: python3 -m pip install --user funasr modelscope soundfile torchaudio")
                sys.exit(1)
            print("[警告] 未找到 FunASR，auto 模式将无法回退")

    success, _, _ = run_command("ffmpeg -version", env=env)
    if not success:
        print("[警告] 未找到 FFmpeg")
    else:
        print("[OK] FFmpeg")

    # 判断是单个视频还是 UP 主主页
    if "space.bilibili.com" in input_url or "/channel/" in input_url:
        print(f"\n[模式] 批量处理 UP 主视频")
        video_urls = get_channel_videos(input_url, limit=args.limit, env=env)

        if not video_urls:
            print("[错误] 未找到视频")
            sys.exit(1)

        # 超过20个视频提示
        if len(video_urls) > 20:
            est_min = len(video_urls) * 3
            est_max = len(video_urls) * 5
            print(f"\n[注意] 共 {len(video_urls)} 个视频，预计耗时 {est_min}-{est_max} 分钟")
            print("[注意] 继续处理...（如需限制数量，使用 --limit 参数）")

        success_count = 0
        for i, url in enumerate(video_urls, 1):
            print(f"\n[进度] {i}/{len(video_urls)}")
            if process_video(url, output_dir, model_size=args.model,
                           compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
                           video_index=i, engine=args.engine,
                           funasr_model=args.funasr_model,
                           funasr_device=args.funasr_device):
                success_count += 1

        # 最终释放模型
        release_model()

        print(f"\n{'='*70}")
        print(f"批量处理完成: {success_count}/{len(video_urls)} 成功")
        print(f"输出目录: {output_dir.absolute()}")
        print(f"{'='*70}")
    else:
        # 补全 BV 号
        if input_url.startswith("BV"):
            input_url = f"https://www.bilibili.com/video/{input_url}/"

        print(f"\n[模式] 处理单个视频")
        process_video(input_url, output_dir, model_size=args.model,
                     compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
                     video_index=1, engine=args.engine,
                     funasr_model=args.funasr_model,
                     funasr_device=args.funasr_device)

        # 释放模型
        release_model()


if __name__ == "__main__":
    main()
