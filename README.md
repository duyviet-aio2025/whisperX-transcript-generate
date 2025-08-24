# 🎙️ WhisperX Transcript Generator

Web application for generating audio transcripts with speaker diarization and video creation.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Install FFmpeg (for video creation)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

### 2. Get HuggingFace Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a token and accept terms for pyannote models

### 3. Run Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## ⚙️ Configuration

### Model Options

- **large-v3**: Best accuracy (recommended)
- **large-v2**: Good accuracy
- **medium**: Balanced speed/accuracy
- **small/base/tiny**: Faster but less accurate

### Compute Types

- **float32**: Best accuracy, more memory
- **float16**: Balanced (GPU only)
- **int8**: Fastest, less memory

### Languages

- **Auto-detect** (recommended)
- **Manual**: vi (Vietnamese), en (English), zh (Chinese), ja (Japanese), etc.

## 📝 Usage

1. **Upload Audio**: WAV, MP3, FLAC, M4A, OGG
2. **Configure**: Set HF token, model, language in sidebar
3. **Generate Transcript**: Click button and wait
4. **Create Video** (optional): Configure and generate MP4 with subtitles
5. **Download**: JSON, TXT, CSV, MP4, ASS files

## 🎬 Video Settings

- **Resolution**: 1920x540, 1280x360, 854x240, 640x180
- **Background**: Dark Blue, Black, Gray, Navy, Purple, Custom
- **Font**: Size 16-72px (default 48), colors: white, yellow, cyan, etc.
- **Subtitles**: Centered in video, ASS format with speaker labels

## 🔧 Troubleshooting

- **CUDA Memory**: Reduce batch size or use smaller model
- **Speaker Diarization**: Check HF token and accept pyannote terms
- **Video Creation**: Install FFmpeg and check `ffmpeg -version`
- **Slow Processing**: Use GPU, smaller model, or increase batch size
