import librosa
import soundfile as sf
import subprocess
import os
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from tqdm import tqdm

def enhance_audio_demucs(input_path, output_path):
    """
    使用 Demucs 进行 AI 降噪 (GPU加速)
    """
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device.upper()}")
    if device == 'cpu':
        print("⚠️  警告: GPU未启用，处理会非常慢！")
    
    # 1. 提取音频
    temp_audio = "temp_audio.wav"
    print("正在提取音频...")
    subprocess.run([
        "ffmpeg", "-i", input_path, "-q:a", "9", "-y", temp_audio
    ], capture_output=True)
    
    # 2. 加载模型
    print("正在加载 Demucs 模型...")
    model = get_model('htdemucs')
    model = model.to(device)
    
    # 3. 加载音频
    print("正在加载音频...")
    wav, sr = librosa.load(temp_audio, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = wav[None, :]
    wav = torch.from_numpy(wav).float().to(device)
    
    # 4. 分离人声（带进度条）
    print("正在分离音源 (GPU加速)...")
    with torch.no_grad():
        stems = apply_model(model, wav[None])[0]
    
    vocals = stems[3].cpu().numpy().T
    
    # 5. 保存
    print("正在保存...")
    sf.write(output_path, vocals, sr)
    os.remove(temp_audio)
    print(f"✓ 处理完成: {output_path}")

if __name__ == "__main__":
    enhance_audio_demucs(
        "media/16教师流.mp4",
        "media/16教师流_enhanced.wav"
    )