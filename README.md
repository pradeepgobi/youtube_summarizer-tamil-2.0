# YouTube Video Summarizer Tamil 2.0

A powerful Streamlit web application that downloads YouTube videos, transcribes them using OpenAI's Whisper, and provides AI-powered summaries. Specially optimized for Tamil content with enhanced NLP processing.

## ✨ Features

- **YouTube Video Processing**: Download videos from YouTube URLs
- **Multi-format Audio Support**: Supports MP3, WAV, M4A audio files
- **Advanced Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **AI-Powered Summaries**: Generates intelligent summaries using Hugging Face transformers
- **Tamil Language Support**: Enhanced processing for Tamil content
- **User-Friendly Interface**: Clean Streamlit web interface
- **Progress Tracking**: Real-time progress indicators
- **Error Handling**: Robust error management and user feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pradeepgobi/youtube_summarizer-tamil-2.0.git
   cd youtube_summarizer-tamil-2.0
   ```

2. **Set up Python environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg:**
   
   **Windows:**
   - Download from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
   - Extract the executables (ffmpeg.exe, ffplay.exe, ffprobe.exe)
   - Place them in the `ffmpeg/bin/` directory
   - Or install using package managers:
     ```bash
     # Using Chocolatey
     choco install ffmpeg
     
     # Using Scoop
     scoop install ffmpeg
     ```
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```

5. **Install Whisper:**
   ```bash
   pip install -U openai-whisper
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage

1. **Start the application** using the command above
2. **Enter a YouTube URL** or upload an audio file
3. **Select processing options** (transcription model, summary length, etc.)
4. **Click "Process"** and wait for the results
5. **View transcription and summary** in the interface
6. **Download results** as text files if needed

## 🛠️ Configuration

### Whisper Models

Choose from different Whisper models based on your needs:

- `tiny`: Fastest, least accurate (~39M parameters)
- `base`: Good balance of speed and accuracy (~74M parameters)
- `small`: Better accuracy (~244M parameters) - **Recommended**
- `medium`: High accuracy (~769M parameters)
- `large`: Best accuracy (~1550M parameters)

### GPU Acceleration

For faster processing, ensure you have CUDA installed:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
youtube_summarizer-tamil-2.0/
├── app.py                    # Main Streamlit application
├── youtube_video_summarizer.py  # Core processing logic
├── tamil_nlp_utils.py       # Tamil language utilities
├── requirements.txt         # Python dependencies
├── complete_requirements.txt # Extended dependencies
├── .streamlit/             # Streamlit configuration
├── images/                 # UI assets
└── ffmpeg/                 # FFmpeg binaries (download separately)
    ├── bin/                # Place FFmpeg executables here
    └── presets/           # FFmpeg presets
```

## 🔧 Dependencies

### Core Requirements
- `streamlit` - Web interface
- `openai-whisper` - Speech recognition
- `yt-dlp` - YouTube video download
- `transformers` - AI summarization
- `torch` - Deep learning framework
- `ffmpeg-python` - Audio processing

### Optional Dependencies
- `cuda` - GPU acceleration
- `scipy` - Scientific computing
- `librosa` - Audio analysis

## 🌟 Advanced Features

### Tamil Language Processing
The application includes specialized processing for Tamil content:
- Enhanced tokenization
- Language-specific preprocessing
- Improved accuracy for Tamil speech recognition

### Batch Processing
Process multiple videos by providing a list of URLs (feature in development).

### Custom Models
Support for custom Whisper models and fine-tuned versions.

## ⚙️ Setup Notes

### FFmpeg Installation
Since FFmpeg binaries are large (500MB+), they are not included in the repository. You need to:

1. Download FFmpeg from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the following files to `ffmpeg/bin/`:
   - `ffmpeg.exe`
   - `ffplay.exe`  
   - `ffprobe.exe`

### Whisper Model Download
Whisper models will be automatically downloaded on first use. Ensure you have sufficient disk space:
- tiny: ~39 MB
- base: ~74 MB
- small: ~244 MB
- medium: ~769 MB
- large: ~1550 MB

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Hugging Face Transformers](https://huggingface.co/transformers/) for summarization
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [Streamlit](https://streamlit.io/) for the web interface

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/pradeepgobi/youtube_summarizer-tamil-2.0/issues) page
2. Create a new issue with detailed information
3. Contact: pradeepgobi@example.com

## 📈 Roadmap

- [ ] Batch processing support
- [ ] Multiple language support
- [ ] Custom model training
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Cloud deployment options

---

**Made with ❤️ for the Tamil community**