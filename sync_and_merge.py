"""Audio-based sync helper for two videos.

Steps this script performs:
1) Extract mono 48 kHz WAV from both videos via ffmpeg.
2) Compute cross-correlation to estimate offset between audios.
3) Print the offset and a ready-to-run ffmpeg command template to stack videos horizontally and keep both audio tracks.

Dependencies:
- ffmpeg installed and on PATH (https://ffmpeg.org/download.html)
- Python packages: numpy, scipy, soundfile (pip install numpy scipy soundfile)
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
import shutil

import numpy as np
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt


def resolve_ffmpeg_bin(user_bin: str | None) -> str:
    """Pick ffmpeg: user-provided path (validated), PATH, or imageio-ffmpeg fallback."""
    if user_bin:
        p = Path(user_bin)
        if p.is_file():
            return str(p)
        found = shutil.which(user_bin)
        if found:
            return user_bin
        print(f"Warning: ffmpeg not found at '{user_bin}'. Falling back to PATH/imageio-ffmpeg.")
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.")


def run_ffmpeg_extract(src: Path, dst: Path, sr: int = 48_000, ffmpeg_bin: str = "ffmpeg") -> None:
    """Extract audio to mono WAV using ffmpeg."""
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-vn",
        "-map",
        "0:a:0",
        "-loglevel",
        "error",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def bandpass(x: np.ndarray, sr: int, low: float = 100.0, high: float = 4000.0) -> np.ndarray:
    nyq = 0.5 * sr
    b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def gcc_phat_lag(ref: np.ndarray, sync: np.ndarray) -> int:
    n = ref.size + sync.size
    R = np.fft.rfft(ref, n=n) * np.conj(np.fft.rfft(sync, n=n))
    R /= np.abs(R) + 1e-12
    corr = np.fft.irfft(R, n=n)
    lags = np.arange(-sync.size + 1, ref.size + 1)
    max_idx = int(np.argmax(corr))
    print(f"[gcc-phat] max at idx={max_idx}, lag={lags[max_idx]}, max={corr[max_idx]:.3f}, left={corr[0]:.3f}, right={corr[-1]:.3f}")
    return int(lags[max_idx])


def compute_offset_seconds(
    ref_wav: Path, sync_wav: Path, method: str = "xcorr", use_bandpass: bool = False,
    max_seconds: int = 300, plot_waveform: bool = False
) -> float:
    """Return offset where positive means sync_wav starts later than ref_wav."""
    ref_audio, sr_ref = sf.read(ref_wav, dtype="float32")
    sync_audio, sr_sync = sf.read(sync_wav, dtype="float32")
    if sr_ref != sr_sync:
        raise ValueError(f"Sample rate mismatch: {sr_ref} vs {sr_sync}")

    # 可选带通
    if use_bandpass:
        ref_audio = bandpass(ref_audio, sr_ref)
        sync_audio = bandpass(sync_audio, sr_sync)

    # 新算法：滑窗点积互相关
    offset = sliding_cross_correlation_offset(
        ref_audio, sync_audio, sr_ref,
        target_sr=2000, max_offset_s=30, compare_sec=60, plot=plot_waveform
    )
    print(f"[sliding xcorr] Estimated offset (sync - ref): {offset:.3f} seconds")
    return offset


def build_ffmpeg_command(
    ref_video: Path, sync_video: Path, offset_sec: float, output: Path, ffmpeg_bin: str = "ffmpeg",
    gpu: str = "none", height: int = 720
) -> list[str]:
    """Return ffmpeg CLI for horizontal stack with both audios preserved."""
    if offset_sec >= 0:
        ref_delay = offset_sec
        sync_delay = 0.0
    else:
        ref_delay = 0.0
        sync_delay = -offset_sec

    cmd: list[str] = [ffmpeg_bin, "-y"]

    # Input 0
    cmd += ["-itsoffset", f"{ref_delay:.3f}"]
    # 不要加 -hwaccel
    cmd += ["-i", str(ref_video)]

    # Input 1
    cmd += ["-itsoffset", f"{sync_delay:.3f}"]
    # 不要加 -hwaccel
    cmd += ["-i", str(sync_video)]

    filter_complex = (
        f"[0:v]scale=-1:{height},setpts=PTS-STARTPTS[v0];"
        f"[1:v]scale=-1:{height},setpts=PTS-STARTPTS[v1];"
        f"[v0][v1]hstack=shortest=1[v];"
        f"[0:a]afftdn[a0];"
        f"[1:a]anlmdn[a1]"
    )
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a0]", "-map", "[a1]",
        "-metadata:s:a:0", "title=Audio-0 (ref,denoised)",
        "-metadata:s:a:1", "title=Audio-1 (sync,denoised)",
        "-disposition:a:0", "default"
    ]

    # Video encoder selection
    if gpu == "nvidia":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    elif gpu == "intel":
        cmd += ["-c:v", "h264_qsv", "-global_quality", "23"]
    elif gpu == "amd":
        cmd += ["-c:v", "h264_amf", "-quality", "quality", "-rc", "vbr", "-q", "23"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]

    cmd += ["-c:a", "aac", "-movflags", "+faststart", str(output)]
    return cmd


def downsample_mono(audio: np.ndarray, orig_sr: int, target_sr: int = 2000) -> np.ndarray:
    """简单抽样法降采样到 target_sr，仅取单声道"""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    ratio = orig_sr / target_sr
    new_len = int(len(audio) / ratio)
    idx = (np.arange(new_len) * ratio).astype(int)
    return audio[idx]


def sliding_cross_correlation_offset(
    ref_audio: np.ndarray, sync_audio: np.ndarray, orig_sr: int,
    target_sr: int = 2000, max_offset_s: int = 30, compare_sec: int = 60, plot: bool = False
) -> float:
    """滑窗点积互相关，返回sync相对ref的最佳偏移（秒，正值表示sync应延后）"""
    # 1. 降采样
    ref_ds = downsample_mono(ref_audio, orig_sr, target_sr)
    sync_ds = downsample_mono(sync_audio, orig_sr, target_sr)
    compare_len = min(len(ref_ds), len(sync_ds), compare_sec * target_sr)
    ref_ds = ref_ds[:compare_len]
    sync_ds = sync_ds[:compare_len]
    max_offset = int(max_offset_s * target_sr)
    best_offset = 0
    max_corr = -np.inf

    offsets = range(-max_offset, max_offset + 1)
    corrs = []
    for lag in offsets:
        if lag < 0:
            a = ref_ds[-lag:compare_len]
            b = sync_ds[:compare_len+lag]
        else:
            a = ref_ds[:compare_len-lag]
            b = sync_ds[lag:compare_len]
        if len(a) == 0 or len(b) == 0:
            corrs.append(0)
            continue
        corr = np.dot(a, b) / len(a)
        corrs.append(corr)
        if corr > max_corr:
            max_corr = corr
            best_offset = lag

    if plot:
        t = np.arange(compare_len) / target_sr
        plt.figure(figsize=(12, 5))
        plt.plot(t, ref_ds, label="ref_ds")
        plt.plot(t, sync_ds, label="sync_ds", alpha=0.7)
        plt.title(f"Downsampled waveform ({target_sr}Hz, {compare_sec}s)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(np.array(list(offsets)) / target_sr, corrs)
        plt.title("Cross-correlation vs offset (s)")
        plt.xlabel("Offset (s)")
        plt.ylabel("Correlation")
        plt.axvline(best_offset / target_sr, color='r', linestyle='--', label=f'Best offset: {best_offset/target_sr:.3f}s')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 注意：返回-ref，和TS实现一致（sync相对ref的校正量）
    return -best_offset / target_sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync two videos by audio")
    parser.add_argument("ref_video", type=Path, help="Reference video (better audio)")
    parser.add_argument("sync_video", type=Path, help="Video to sync against reference")
    parser.add_argument("--output", type=Path, default=Path("merged.mkv"), help="Output file")
    parser.add_argument("--ffmpeg", type=str, default=None, help="Path to ffmpeg executable")
    parser.add_argument("--method", choices=["xcorr", "gcc-phat"], default="xcorr", help="Offset estimation method")
    parser.add_argument("--bandpass", action="store_true", help="Apply 100–4000 Hz band-pass before correlation")
    parser.add_argument("--gpu", choices=["none", "nvidia", "intel", "amd"], default="none", help="Use GPU encoding/decoding")
    parser.add_argument("--height", type=int, default=720, help="Per-stream output height before hstack")
    parser.add_argument("--plot", action="store_true", help="Plot first 120s waveform for both audios")  # <-- 加这一行
    args = parser.parse_args()

    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        ref_wav = tmpdir_path / "ref.wav"
        sync_wav = tmpdir_path / "sync.wav"

        run_ffmpeg_extract(args.ref_video, ref_wav, ffmpeg_bin=ffmpeg_bin)
        run_ffmpeg_extract(args.sync_video, sync_wav, ffmpeg_bin=ffmpeg_bin)

        offset_sec = compute_offset_seconds(ref_wav, sync_wav, method=args.method, use_bandpass=args.bandpass, plot_waveform=True)
        print(f"Estimated offset (sync - ref): {offset_sec:.3f} seconds")
        cmd = build_ffmpeg_command(
            args.ref_video, args.sync_video, offset_sec, args.output, ffmpeg_bin=ffmpeg_bin, gpu=args.gpu, height=args.height
        )
        print("\nSuggested ffmpeg command:")
        print(" ".join(cmd))

        # 构造适合 PowerShell 的 ffmpeg 命令字符串（加引号）
        def quote_arg(arg):
            # 如果参数包含空格或特殊字符，则加英文双引号
            if any(c in arg for c in ' []();='):
                return f'"{arg}"'
            return arg

        cmd_str = " ".join(quote_arg(str(x)) for x in cmd)
        print("\nPowerShell-ready ffmpeg command (copy & paste to run):")
        print(cmd_str)
        print("\nNotes:\n- Positive offset means the sync_video starts later; we delay the reference video to align.\n- Output keeps both audio tracks in MKV so you can pick during playback.")


if __name__ == "__main__":
    main()
