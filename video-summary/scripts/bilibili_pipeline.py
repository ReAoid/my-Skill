#!/usr/bin/env python3
"""
B站视频批量转写 Pipeline
功能：字幕优先提取 → 无字幕时下载音频 → Whisper 转写 → 生成 TXT
支持：单个视频、UP主批量、断点续传
macOS：自动检测 Apple Silicon，优先用 mlx-whisper (Metal GPU)
Windows：使用 faster-whisper (CUDA/CPU)

规范：强制使用 Anaconda/Miniconda 虚拟环境，启动时自动校验。
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
import time
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
CONDA_ENV_NAME = "bilibili_trans"
LONG_VIDEO_THRESHOLD = 900  # 15分钟
BATCH_CONFIRM_THRESHOLD = 20

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


# ============ P0: Conda 环境检测 ============

def check_conda_env():
    """检测 Conda 环境，强制使用 Anaconda/Miniconda 虚拟环境。"""
    conda_exe = shutil.which("conda") or shutil.which("conda.exe")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    print("=" * 70)
    print("  [环境校验] Conda 虚拟环境检测")
    print("=" * 70)

    if not conda_exe:
        print("  [警告] 未检测到 conda 命令！")
        print("  本工具仅支持 Anaconda/Miniconda 虚拟环境运行。")
        print()
        _print_conda_install_guide()
        print()
        print("  请安装 Miniconda 后，创建并激活虚拟环境再运行本脚本。")
        print("=" * 70)
        print()
        return False

    print(f"  [OK] conda 已安装: {conda_exe}")

    if not conda_env:
        print("  [警告] 当前未激活 conda 虚拟环境！")
        print(f"  请先激活 {CONDA_ENV_NAME} 环境再运行：")
        _print_conda_activate_guide()
        print()
        print("  或使用以下命令一键创建并激活环境：")
        _print_conda_create_guide()
        print("=" * 70)
        print()
        return False

    if conda_env != CONDA_ENV_NAME:
        print(f"  [警告] 当前 conda 环境为 '{conda_env}'，推荐使用 '{CONDA_ENV_NAME}'")
        print(f"  请切换环境: conda activate {CONDA_ENV_NAME}")
        print("=" * 70)
        print()
        # 不强制退出，只是警告
    else:
        print(f"  [OK] 当前 conda 环境: {conda_env}")

    print()
    return True


def _print_conda_install_guide():
    """输出 Conda 安装指引（区分系统）。"""
    system = platform.system()
    print("  ┌─ Miniconda 安装指引 ──────────────────────────────")
    if system == "Windows":
        print("  │ 1. 下载安装包:")
        print("  │    https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe")
        print("  │ 2. 安装时勾选 'Add Miniconda3 to my PATH'")
        print("  │ 3. 重启终端，验证: conda --version")
    elif system == "Darwin":
        arch = platform.machine()
        if arch == "arm64":
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else:
            url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        print(f"  │ wget {url}")
        print(f"  │ bash Miniconda3-latest-MacOSX-*.sh")
        print("  │ conda init zsh")
        print("  │ source ~/.zshrc")
    else:
        print("  │ 请访问 https://docs.conda.io/en/latest/miniconda.html 下载安装")
    print("  └──────────────────────────────────────────────────")


def _print_conda_create_guide():
    """输出创建 conda 虚拟环境的命令。"""
    system = platform.system()
    python_cmd = "python" if system == "Windows" else "python3"
    print(f"  ┌─ 一键创建 {CONDA_ENV_NAME} 环境 ──────────────────────")
    print(f"  │ conda create -n {CONDA_ENV_NAME} python=3.10 -y")
    print(f"  │ conda activate {CONDA_ENV_NAME}")
    print(f"  │ conda install ffmpeg -y")
    print(f"  │ pip install yt-dlp opencc-python-reimplemented")
    print()
    if system == "Windows":
        print("  │ # CUDA 显卡:")
        print("  │ pip install torch==2.3.1+cu121 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121")
        print("  │ pip install faster-whisper funasr modelscope soundfile")
        print("  │")
        print("  │ # 纯 CPU:")
        print("  │ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("  │ pip install faster-whisper funasr modelscope soundfile")
    elif system == "Darwin" and platform.machine() == "arm64":
        print("  │ # Apple Silicon:")
        print("  │ pip install mlx-whisper torch torchvision torchaudio funasr modelscope soundfile")
    else:
        print("  │ # CPU/Intel:")
        print("  │ pip install faster-whisper torch torchvision torchaudio funasr modelscope soundfile")
    print("  └──────────────────────────────────────────────────")


def _print_conda_activate_guide():
    """输出 conda 环境激活命令。"""
    system = platform.system()
    if system == "Windows":
        print(f"  > conda activate {CONDA_ENV_NAME}")
    else:
        print(f"  $ conda activate {CONDA_ENV_NAME}")


# ============ P0: Cookie 自动加载 ============

def check_and_load_cookie():
    """自动检测 ./cookie.txt，存在则注入 yt-dlp 环境变量；缺失则打印配置步骤。"""
    cookie_path = Path.cwd() / "cookie.txt"
    if cookie_path.exists():
        os.environ["YT_DLP_COOKIE"] = str(cookie_path)
        print(f"  [OK] 已自动加载 Cookie: {cookie_path}")
        return True

    # 也检查脚本同级目录
    script_dir = Path(__file__).resolve().parent
    cookie_path2 = script_dir / "cookie.txt"
    if cookie_path2.exists():
        os.environ["YT_DLP_COOKIE"] = str(cookie_path2)
        print(f"  [OK] 已自动加载 Cookie: {cookie_path2}")
        return True

    print("  [注意] 未检测到 cookie.txt 文件")
    print("  ┌─ Cookie 配置步骤（B站防403限制）───────────────")
    print("  │ 1. 浏览器安装插件: 'Get Cookies LOCAL'")
    print("  │ 2. 打开 https://www.bilibili.com")
    print("  │ 3. 点击插件图标，导出 cookies 为 txt 格式")
    print("  │ 4. 将文件重命名为 cookie.txt")
    print("  │ 5. 放置到脚本同级目录")
    print("  │ 6. 脚本启动自动加载，无需传入 --cookie 参数")
    print("  └──────────────────────────────────────────────")
    return False


# ============ P0: 硬件检测与自动适配 ============

def detect_hardware_and_optimize(args):
    """检测硬件环境，自动适配最优运行参数，返回优化后的 args。"""
    system = platform.system()

    # 检测 Apple Silicon
    if system == "Darwin" and platform.machine() == "arm64":
        print("  [硬件] Apple Silicon (M系列芯片) 检测通过")
        # Apple Silicon 锁定 mlx-whisper
        if args.engine == DEFAULT_ENGINE:
            pass  # whisper 模式会自动选 mlx-whisper

    # Windows 检测 CUDA
    if system == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                print(f"  [硬件] NVIDIA GPU: {torch.cuda.get_device_name(0)}")

                # GTX 16xx 老显卡检测
                if any(k in gpu_name for k in ['gtx 16', 'gtx 165', 'gtx 166', 'gtx105', 'gtx 10']):
                    print("  [优化] 检测到老架构显卡，自动优化参数:")
                    print("         - 强制关闭 FP16（老显卡FP16性能极弱）")
                    print("         - 使用 int8 量化提速")
                    args.fp16 = False
                else:
                    # 新显卡
                    print("  [优化] 检测到新架构显卡，可使用 medium/large 大模型")
            else:
                print("  [硬件] CUDA 不可用，将使用 CPU（建议使用 tiny/base 轻量模型）")
        except ImportError:
            print("  [硬件] torch 未安装，无法检测 CUDA")
        except Exception:
            pass

    # 默认开启 cleanup-audio
    if not hasattr(args, 'cleanup_audio') or not args.cleanup_audio:
        args.cleanup_audio = True

    # 默认模型固定 small
    if not hasattr(args, 'model') or args.model is None:
        args.model = DEFAULT_MODEL

    return args


def print_hardware_tips():
    """根据硬件检测结果输出运行优化建议。"""
    system = platform.system()
    print("  ┌─ 硬件运行优化建议 ──────────────────────────────")

    if system == "Darwin" and platform.machine() == "arm64":
        print("  │ ● Apple Silicon: 推荐使用 mlx-whisper (Metal GPU 加速)")
        print("  │ ● 无需安装 faster-whisper (ctranslate2 不支持 Apple GPU)")
        print("  │ ● FunASR 默认走 MPS 加速")
        print("  │ ● 转写 30 分钟视频约 3-4 分钟")
    elif system == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).lower()
                if any(k in gpu_name for k in ['gtx 16', 'gtx 165', 'gtx 166', 'gtx105', 'gtx 10']):
                    print("  │ ● GTX 老显卡: 自动关闭 FP16，使用 int8 量化")
                    print("  │ ● 建议使用 small/base 模型")
                    print("  │ ● 避免使用 large 模型（显存不足）")
                else:
                    print("  │ ● 新架构 GPU: 可使用 medium/large 大模型")
                    print("  │ ● CUDA 加速已启用")
            else:
                print("  │ ● 纯 CPU 环境: 建议使用 tiny/base 轻量化模型")
                print("  │ ● 预计转写速度比 GPU 慢 10-20 倍")
                print("  │ ● 长视频建议分批处理")
        except ImportError:
            print("  │ ● 建议安装 torch 以检测 GPU 状态")
    else:
        print("  │ ● 通用建议: 根据 CPU 性能选择合适的模型大小")

    print("  │ ● 长视频（>15分钟）建议在本地终端运行，避免进程超时被 kill")
    print("  └──────────────────────────────────────────────")


# ============ P0: 启动全量环境自检汇总 ============

def print_env_summary():
    """启动时一次性输出完整环境状态汇总面板。"""
    system = platform.system()
    arch = platform.machine()

    print()
    print("=" * 70)
    print("  [环境汇总] B站视频转写 Pipeline - 系统状态面板")
    print("=" * 70)

    # 系统信息
    print(f"  [系统] {system} / {arch}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "未激活")
    print(f"  [Conda] 环境: {conda_env}")

    # yt-dlp
    success, stdout, _ = run_command("yt-dlp --version", timeout=10)
    if success:
        ver = stdout.strip().split('\n')[0] if stdout else "?"
        print(f"  [yt-dlp] {ver} [OK]")
    else:
        print(f"  [yt-dlp] 未安装 [错误]")

    # ffmpeg
    env = get_ffmpeg_env()
    success, stdout, _ = run_command("ffmpeg -version", env=env, timeout=10)
    if success:
        ver = stdout.split('\n')[0] if stdout else "?"
        print(f"  [ffmpeg] {ver} [OK]")
    else:
        print(f"  [ffmpeg] 未找到 [警告]")

    # 硬件加速
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        mps_avail = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        if cuda_avail:
            print(f"  [CUDA] {torch.cuda.get_device_name(0)} [OK]")
        elif mps_avail:
            print(f"  [MPS] Apple Metal GPU [OK]")
        else:
            print(f"  [加速] 仅 CPU [警告]")
    except ImportError:
        print(f"  [torch] 未安装 [警告]")

    # Whisper 引擎
    if system == "Darwin" and arch == "arm64":
        try:
            import mlx_whisper
            print(f"  [mlx-whisper] 可用 [OK]")
        except ImportError:
            print(f"  [mlx-whisper] 未安装 [警告]")
    else:
        try:
            from faster_whisper import WhisperModel
            print(f"  [faster-whisper] 可用 [OK]")
        except ImportError:
            print(f"  [faster-whisper] 未安装 [警告]")

    # FunASR
    try:
        from funasr import AutoModel
        print(f"  [FunASR] 可用 [OK]")
    except ImportError:
        print(f"  [FunASR] 未安装 [信息]")

    print("=" * 70)
    print()


# ============ P0: 依赖缺失安装指引 ============

def print_dependency_install_guide(missing_dep):
    """根据缺失的依赖和当前操作系统，输出完整 pip 一键安装命令。"""
    system = platform.system()
    is_apple_silicon = (system == "Darwin" and platform.machine() == "arm64")

    guides = {
        "yt-dlp": "pip install yt-dlp",
        "mlx-whisper": "pip install mlx-whisper",
        "faster-whisper": "pip install faster-whisper",
        "funasr": "pip install funasr modelscope soundfile",
        "opencc": "pip install opencc-python-reimplemented",
        "torch": "",
    }

    print(f"  [安装指引] 缺失依赖: {missing_dep}")

    if missing_dep == "torch":
        if system == "Windows":
            try:
                import torch
                if torch.cuda.is_available():
                    print("  pip install torch==2.3.1+cu121 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121")
                else:
                    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            except ImportError:
                print("  # Windows CUDA 显卡:")
                print("  pip install torch==2.3.1+cu121 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121")
                print("  # 或纯 CPU:")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        elif is_apple_silicon:
            print("  pip install torch torchvision torchaudio")
        else:
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    elif missing_dep in guides:
        print(f"  {guides[missing_dep]}")
    else:
        print(f"  pip install {missing_dep}")


# ============ P0: FFmpeg 安装指引 ============

def print_ffmpeg_install_guide():
    """FFmpeg 检索失败时，区分系统输出安装命令和手动部署说明。"""
    system = platform.system()

    print("  ┌─ FFmpeg 安装指引 ──────────────────────────────")
    if system == "Windows":
        print("  │ 方法1 (推荐): winget install ffmpeg")
        print("  │")
        print("  │ 方法2 (便携版):")
        print("  │ 1. 下载 builds.general.works 的 ffmpeg-master-latest-win64-gpl.zip")
        print("  │ 2. 解压到脚本同级目录，保持目录名: ffmpeg-master-latest-win64-gpl")
        print("  │ 3. 脚本会自动扫描该目录")
        print("  │")
        print("  │ 验证: ffmpeg -version")
    elif system == "Darwin":
        print("  │ 方法1 (推荐): brew install ffmpeg")
        print("  │")
        print("  │ 方法2 (手动):")
        print("  │ 1. 下载 https://evermeet.cx/ffmpeg/ 的 ffmpeg 可执行文件")
        print("  │ 2. 放入 /usr/local/bin/ 或 ~/bin/")
        print("  │")
        print("  │ 验证: ffmpeg -version")
    else:
        print("  │ sudo apt install ffmpeg   # Debian/Ubuntu")
        print("  │ sudo yum install ffmpeg   # CentOS/RHEL")
    print("  └──────────────────────────────────────────────")


# ============ P0: 长视频警告 ============

def check_long_video_warning(duration_seconds):
    """单视频时长 > 15分钟，自动打印超时风险提示。"""
    if duration_seconds > LONG_VIDEO_THRESHOLD:
        dur_min = duration_seconds // 60
        dur_sec = duration_seconds % 60
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  [长视频警告]                                          ║")
        print(f"  ║  当前视频时长: {dur_min}分{dur_sec}秒 ({duration_seconds:.0f}s)           ║")
        print("  ║  超过15分钟阈值，转写可能需要较长时间                   ║")
        print("  ║                                                         ║")
        print("  ║  建议: 在本地终端手动运行，避免进程超时被 kill          ║")
        print(f"  ║  python -u bilibili_pipeline.py \"<URL>\"                ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()
        return True
    return False


# ============ P0: 批量交互确认 ============

def interactive_batch_confirm(video_count, estimated_time_min, estimated_time_max):
    """视频数量 > 20 时交互式询问用户是否继续。"""
    if video_count <= BATCH_CONFIRM_THRESHOLD:
        return True

    print()
    print("=" * 70)
    print(f"  [批量确认] 共 {video_count} 个视频")
    print(f"  [耗时预估] 预计 {estimated_time_min}-{estimated_time_max} 分钟")
    print("=" * 70)
    print()

    # 标准输入确认
    try:
        response = input("  是否继续处理? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("  已取消处理。如需限制数量，使用 --limit 参数。")
            return False
        print("  继续处理...")
        return True
    except (EOFError, KeyboardInterrupt):
        print("\n  已取消处理。")
        return False


# ============ P0: 通用工具函数 ============

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


# ============ 转写引擎 ============

def transcribe_with_mlx_whisper(audio_path, model_size=DEFAULT_MODEL):
    """使用 mlx-whisper (Apple Silicon Metal GPU) 转写音频"""
    global _cached_model, _cached_model_key

    import mlx_whisper

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
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"[错误] 加载模型 {model_size} 失败: {e}", flush=True)
            # P1: 模型降级容错
            raise
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
        try:
            model = AutoModel(
                model=model_name,
                vad_model="fsmn-vad",
                punc_model="ct-punc-c",
                disable_update=True,
                disable_pbar=False,
                device=device,
            )
        except Exception as e:
            print(f"[错误] 加载 FunASR 模型 {model_name} 失败: {e}", flush=True)
            # P1: FunASR 加载失败，让上层回退
            raise
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


# ============ P1: 模型双层降级容错 ============

def _try_transcribe_with_fallback(audio_path, model_size, compute_type, engine,
                                    funasr_model, funasr_device):
    """
    转写统一入口，内置双层降级容错：
    1. large/medium 下载失败 → 自动降级 small/tiny
    2. Whisper 全部失败 → 自动回退 FunASR
    3. FunASR 失败 → 自动切回 Whisper
    """
    model_sizes_fallback = ["large", "medium", "small", "tiny"]
    current_models = model_sizes_fallback[model_sizes_fallback.index(model_size):] if model_size in model_sizes_fallback else [model_size]

    # 尝试 Whisper 系列
    if engine in ("whisper", "auto"):
        for try_model in current_models:
            try:
                if is_apple_silicon():
                    data = transcribe_with_mlx_whisper(audio_path, model_size=try_model)
                    method = f"mlx-whisper ({try_model}, Apple Silicon GPU)"
                else:
                    data = transcribe_with_faster_whisper(
                        audio_path, model_size=try_model, compute_type=compute_type
                    )
                    method = f"Faster Whisper ({try_model}, {compute_type})"
                return data, method
            except Exception as e:
                if try_model != current_models[-1]:
                    print(f"[降级] {try_model} 模型失败，自动降级到更小模型: {e}", flush=True)
                else:
                    print(f"[警告] Whisper 全部模型均失败: {e}", flush=True)

        # Whisper 全部失败，auto 模式回退 FunASR
        if engine == "auto":
            print("[回退] Whisper 全部失败，自动回退 FunASR...", flush=True)
            try:
                device = choose_funasr_device(funasr_device)
                data = transcribe_with_funasr(audio_path, model_name=funasr_model, device=device)
                method = f"FunASR fallback ({funasr_model}, {device})"
                return data, method
            except Exception as e:
                print(f"[错误] FunASR 回退也失败: {e}", flush=True)
                raise
        else:
            raise

    # engine == "funasr"
    try:
        device = choose_funasr_device(funasr_device)
        data = transcribe_with_funasr(audio_path, model_name=funasr_model, device=device)
        method = f"FunASR ({funasr_model}, {device})"
        return data, method
    except Exception as e:
        print(f"[警告] FunASR 失败，尝试回退 Whisper: {e}", flush=True)
        # FunASR 失败回退 Whisper
        for try_model in current_models:
            try:
                if is_apple_silicon():
                    data = transcribe_with_mlx_whisper(audio_path, model_size=try_model)
                    method = f"mlx-whisper fallback ({try_model}, Apple Silicon GPU)"
                else:
                    data = transcribe_with_faster_whisper(
                        audio_path, model_size=try_model, compute_type=compute_type
                    )
                    method = f"Faster Whisper fallback ({try_model}, {compute_type})"
                return data, method
            except Exception as e2:
                if try_model == current_models[-1]:
                    print(f"[错误] Whisper 回退也全部失败: {e2}", flush=True)
                    raise
        raise


def transcribe_audio(audio_path, model_size=DEFAULT_MODEL,
                     compute_type=DEFAULT_COMPUTE_TYPE, engine=DEFAULT_ENGINE,
                     funasr_model=DEFAULT_FUNASR_MODEL,
                     funasr_device=DEFAULT_FUNASR_DEVICE):
    """
    转写入口（带双层降级容错）。
    返回 (transcript_data, method)。
    """
    return _try_transcribe_with_fallback(
        audio_path, model_size, compute_type, engine, funasr_model, funasr_device
    )


# ============ 输出 ============

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


# ============ P1: 输出目录自适应规则 ============

def determine_output_dir(args, input_path=None):
    """
    P1: 输出目录自适应规则固化。
    - 本地音视频：默认输出 `原文件名_transcribe` 文件夹
    - B站视频：固定 `./bilibili_output`
    - 用户传 --output-dir 则按用户指定
    """
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        # 检查是否用户显式传入（非默认值）
        return Path(args.output_dir)

    # 本地文件
    if input_path and Path(input_path).exists():
        p = Path(input_path)
        return Path.cwd() / f"{safe_filename(p.stem)}_transcribe"

    # B站视频
    return Path(DEFAULT_OUTPUT_DIR)


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

    # P1: 输出目录自适应
    base_output_dir = Path(output_dir) if output_dir else (
        media_path.parent / f"{safe_filename(media_path.stem)}_transcribe"
    )
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

    # P0: 长视频超时风险提示
    check_long_video_warning(dur)

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


def print_task_summary(start_time, total, success_count, failed_list, output_dir, is_batch=False):
    """P1: 任务结束全局统计汇总输出。"""
    elapsed = time.time() - start_time
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)

    print()
    print("=" * 70)
    print("  [任务统计] 执行完成汇总")
    print("=" * 70)
    print(f"  总视频数:     {total}")
    print(f"  成功:         {success_count}")
    print(f"  失败:         {len(failed_list)}")
    if failed_list:
        print(f"  失败列表:     {', '.join(failed_list)}")
    print(f"  总运行耗时:   {elapsed_min}分{elapsed_sec}秒")
    print(f"  输出目录:     {output_dir.absolute()}")
    print("=" * 70)
    print()


def main():
    global MODEL_RELEASE_INTERVAL, _cached_model, _cached_model_key

    # ========== 启动时间 ==========
    script_start_time = time.time()

    # ========== P0: Conda 环境检测 ==========
    check_conda_env()

    # ========== P0: 启动全量环境自检汇总 ==========
    print_env_summary()

    # ========== P0: Cookie 自动加载 ==========
    check_and_load_cookie()

    # ========== 参数解析 ==========
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
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（默认：B站视频 -> ./bilibili_output，本地文件 -> 原文件名_transcribe）")
    parser.add_argument("--cleanup-audio", action="store_true", default=True,
                        help="转写后删除音频文件（默认开启）")
    parser.add_argument("--no-cleanup-audio", action="store_false", dest="cleanup_audio",
                        help="保留音频文件")
    parser.add_argument("--transcribe-only", help="仅转写指定音频文件（跳过下载）")
    parser.add_argument("--extract-audio", help="仅从本地视频提取音频，不执行转写")
    parser.add_argument("--cookie", help="yt-dlp Cookie 文件路径（如已放置 cookie.txt 则自动加载）")
    parser.add_argument("--release-interval", type=int, default=MODEL_RELEASE_INTERVAL,
                        help=f"每处理N个视频释放模型（默认{MODEL_RELEASE_INTERVAL}）")
    args = parser.parse_args()

    # ========== P0: 硬件自动适配最优参数 ==========
    args = detect_hardware_and_optimize(args)

    MODEL_RELEASE_INTERVAL = args.release_interval

    # ========== P0: 硬件优化建议输出 ==========
    print_hardware_tips()
    print()

    # 优先使用 --cookie 参数，否则函数 check_and_load_cookie() 已在上面触发过
    if args.cookie:
        os.environ["YT_DLP_COOKIE"] = args.cookie

    # ========== 依赖检查 ==========
    print("[检查] 检查依赖...")

    deps_ok = True

    # yt-dlp
    success, _, _ = run_command("yt-dlp --version")
    if not success:
        print("[错误] 未找到 yt-dlp")
        print_dependency_install_guide("yt-dlp")
        deps_ok = False
    else:
        print("[OK] yt-dlp")

    # Whisper 引擎
    if is_apple_silicon():
        try:
            import mlx_whisper
            print("[OK] mlx-whisper (Apple Silicon GPU)")
        except ImportError:
            print("[警告] 未找到 mlx-whisper，将使用 faster-whisper (CPU很慢)")
            print_dependency_install_guide("mlx-whisper")
            print("  建议安装: pip install mlx-whisper")
            # 不阻止运行，降级到 faster-whisper CPU
    else:
        try:
            from faster_whisper import WhisperModel
            print("[OK] faster-whisper")
        except ImportError:
            print("[错误] 未找到 faster-whisper")
            print_dependency_install_guide("faster-whisper")
            deps_ok = False

        try:
            import torch
            if torch.cuda.is_available():
                print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}")
            else:
                print("[警告] CUDA 不可用，将使用 CPU（速度很慢）")
        except ImportError:
            print("[警告] torch 未安装，无法检测 CUDA")
            if not is_apple_silicon():
                print_dependency_install_guide("torch")

    # FunASR
    if args.engine in ("funasr", "auto"):
        try:
            from funasr import AutoModel
            print(f"[OK] FunASR, device: {choose_funasr_device(args.funasr_device)}")
        except ImportError:
            if args.engine == "funasr":
                print("[错误] 未找到 FunASR")
                print_dependency_install_guide("funasr")
                deps_ok = False
            else:
                print("[警告] 未找到 FunASR，auto 模式将无法回退")
                print_dependency_install_guide("funasr")

    # FFmpeg
    env = get_ffmpeg_env()
    success, _, _ = run_command("ffmpeg -version", env=env)
    if not success:
        print("[警告] 未找到 FFmpeg，部分功能将受限")
        print_ffmpeg_install_guide()
    else:
        print("[OK] FFmpeg")

    if not deps_ok and not args.url:
        print("\n[错误] 核心依赖缺失，请先安装所需依赖后重试。")
        sys.exit(1)

    print()

    # ========== 判断执行模式 ==========
    if not args.url:
        print()
        print("=" * 70)
        print("  B站视频批量转写 Pipeline")
        print("=" * 70)
        print()
        print("  用法: python bilibili_pipeline.py <视频URL或UP主主页URL> [选项]")
        print()
        print("  示例:")
        print("    python bilibili_pipeline.py https://www.bilibili.com/video/BV1xx411c7mD")
        print("    python bilibili_pipeline.py https://space.bilibili.com/123456 --limit 10")
        print("    python bilibili_pipeline.py 本地视频.mp4")
        print("    python bilibili_pipeline.py 本地音频.m4a")
        print()
        print("  选项:")
        print("    --limit N           仅处理前 N 个视频")
        print("    --model small       模型大小: tiny/base/small/medium/large")
        print("    --engine whisper    转写引擎: whisper/funasr/auto")
        print("    --output-dir DIR    指定输出目录")
        print("    --no-cleanup-audio  保留音频文件")
        print("    --cookie FILE       Cookie 文件路径")
        print("    --transcribe-only   仅转写已有音频文件")
        print("    --extract-audio     仅提取音频，不转写")
        print()
        sys.exit(0)

    input_url = args.url

    # 本地音视频路径：自动选择本地处理流程
    local_path = Path(input_url).expanduser()
    if local_path.exists():
        # P1: 输出目录自适应
        output_dir = determine_output_dir(args, input_path=input_url)
        success = process_local_media(
            local_path, output_dir=output_dir, model_size=args.model,
            compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
            engine=args.engine, funasr_model=args.funasr_model,
            funasr_device=args.funasr_device
        )
        # P1: 任务结束统计
        if success:
            print_task_summary(
                script_start_time, 1, 1, [], output_dir, is_batch=False
            )
        sys.exit(0 if success else 1)

    # 判断是单个视频还是 UP 主主页
    if "space.bilibili.com" in input_url or "/channel/" in input_url:
        print(f"\n[模式] 批量处理 UP 主视频")
        video_urls = get_channel_videos(input_url, limit=args.limit, env=env)

        if not video_urls:
            print("[错误] 未找到视频")
            sys.exit(1)

        # P1: 输出目录自适应
        output_dir = determine_output_dir(args)

        # P0: 批量交互确认
        video_count = len(video_urls)
        est_min = video_count * 3
        est_max = video_count * 5

        if not interactive_batch_confirm(video_count, est_min, est_max):
            sys.exit(0)

        success_count = 0
        failed_list = []
        for i, url in enumerate(video_urls, 1):
            print(f"\n[进度] {i}/{video_count}")
            if process_video(url, output_dir, model_size=args.model,
                           compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
                           video_index=i, engine=args.engine,
                           funasr_model=args.funasr_model,
                           funasr_device=args.funasr_device):
                success_count += 1
            else:
                # 提取 BV 号
                bv_match = re.search(r'(BV[a-zA-Z0-9]+)', url)
                if bv_match:
                    failed_list.append(bv_match.group(1))

        # 最终释放模型
        release_model()

        # P1: 任务结束全局统计汇总
        print_task_summary(script_start_time, video_count, success_count,
                          failed_list, output_dir, is_batch=True)

    else:
        # 补全 BV 号
        if input_url.startswith("BV"):
            input_url = f"https://www.bilibili.com/video/{input_url}/"

        print(f"\n[模式] 处理单个视频")

        # P1: 输出目录自适应
        output_dir = determine_output_dir(args)

        success = process_video(input_url, output_dir, model_size=args.model,
                               compute_type=args.compute_type, cleanup_audio=args.cleanup_audio,
                               video_index=1, engine=args.engine,
                               funasr_model=args.funasr_model,
                               funasr_device=args.funasr_device)

        # 释放模型
        release_model()

        # P1: 任务结束统计
        failed_list = []
        if not success:
            bv_match = re.search(r'(BV[a-zA-Z0-9]+)', input_url)
            if bv_match:
                failed_list.append(bv_match.group(1))

        print_task_summary(script_start_time, 1, 1 if success else 0,
                          failed_list, output_dir, is_batch=False)

    print(f"[完成] 所有任务执行完毕")


if __name__ == "__main__":
    main()
