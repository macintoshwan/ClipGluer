# ClipGluer

音频自动对齐与视频拼接工具。  
支持两路视频音频自动检测偏移、降噪、横向拼接输出。

## 依赖

- Python 3.8+
- ffmpeg (需加入 PATH)
- pip install -r requirements.txt

## 用法

```bash
python sync_and_merge.py <ppt流.mp4> <教师流.mp4> --output merged.mkv --gpu nvidia --bandpass --plot
```

- 自动输出 PowerShell 可用的 ffmpeg 合成命令。
- 支持音频降噪、波形可视化。

## 说明

- 输出文件 merged.mkv，包含两路音轨和横向拼接视频。
- 推荐用 PotPlayer/VLC 播放，可切换音轨。