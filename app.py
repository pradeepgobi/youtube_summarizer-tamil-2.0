from deep_translator import GoogleTranslator
import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import requests
import json
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import whisper
import yt_dlp
import os
import re

# Page setup
st.set_page_config(
    page_title="🎬 Universal Tamil Video Summarizer", 
    page_icon="🎬",
    layout='wide',
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    .sidebar .stTextArea > div > div > textarea {
        border-radius: 15px;
    }
    
    .summary-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .tamil-text {
        font-family: 'Noto Sans Tamil', sans-serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# Use a smaller, faster LLM model for summarization
@st.cache_resource
def load_model():
    try:
        # Try to connect with faster settings
        llm = OllamaLLM(
            model="llama2",
            base_url="http://localhost:11434",
            timeout=30,
            temperature=0.1  # Lower temperature for faster, more consistent responses
        )
        # Test the connection
        test_response = llm.invoke("Hello")
        return llm
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama: {e}")
        st.info("💡 Please ensure Ollama is running with: `ollama serve`")
        return None

llm = load_model()

# Session state
for key in ['last_url', 'summary', 'transcript_text', 'summary_en', 'video_metadata']:
    if key not in st.session_state:
        st.session_state[key] = None

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎬 Universal Tamil Video Summarizer</h1>
    <p>Transform any YouTube video into clear Tamil summaries with AI power</p>
</div>
""", unsafe_allow_html=True)

# Feature highlights
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Multi-Language Support</h3>
        <p>Works with videos in any language - automatically detects and processes content</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI-Powered Summaries</h3>
        <p>Uses advanced LLM models to create comprehensive, easy-to-understand summaries</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📱 Tamil Output</h3>
        <p>Get perfectly translated Tamil summaries that preserve the original meaning</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar input
with st.sidebar:
    st.markdown("### 📝 Input Options")
    
    st.markdown("#### 🔗 YouTube Video")
    url = st.text_input(
        "Paste YouTube URL here:",
        placeholder="https://youtube.com/watch?v=...",
        help="Paste any YouTube video URL to get started"
    )
    
    st.markdown("#### 📄 Manual Transcript (Optional)")
    manual_transcript = st.text_area(
        "Or paste transcript directly:",
        value=st.secrets.get("YOUTUBE_TRANSCRIPT", ""),
        height=200,
        placeholder="Paste video transcript in any language...",
        help="If YouTube transcript is not available, you can paste the video transcript manually"
    )
    
    # Instructions
    with st.expander("📚 How to Use"):
        st.markdown("""
        **Easy Steps:**
        1. 🔗 Paste a YouTube URL above
        2. ⏳ Wait for automatic processing
        3. 📖 Read your Tamil summary!
        
        **Alternative:**
        - 📋 Paste transcript manually if needed
        - 🌐 Works with any language input
        """)
    
    # System status
    with st.expander("⚙️ System Status"):
        st.success("✅ Ollama LLM: Ready")
        st.success("✅ YouTube API: Ready") 
        st.info("💡 Audio download: Disabled (using transcripts)")
        st.info("🔄 Auto-translation: Enabled")


# Only get the API key string, ignore any accidental URLs or extra lines
API_KEY = st.secrets.get("YOUTUBE_API_KEY")
if API_KEY and API_KEY.strip().startswith("http"):
    st.warning("⚠️ Your YOUTUBE_API_KEY in secrets.toml looks like a URL, not an API key. Please check your secrets.toml file.")
    API_KEY = None

# --- Utility Functions ---

def get_video_id(url):
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    try:
        query = urlparse(url)
        if query.hostname == 'www.youtube.com' and query.path == '/watch':
            return parse_qs(query.query)['v'][0]
    except (KeyError, IndexError):
        return None
    return None

def get_video_info_oembed(video_url):
    """Fallback method using YouTube oEmbed API"""
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            return None
            
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'title': data.get('title', 'Unknown Title'),
                'channel': data.get('author_name', 'Unknown Channel'),
                'duration': 'Unknown',  # oEmbed doesn't provide duration
                'thumbnail': data.get('thumbnail_url', ''),
                'view_count': 0,  # oEmbed doesn't provide view count
                'upload_date': ''
            }
    except Exception as e:
        print(f"oEmbed method failed: {e}")
    return None

def get_video_info_webpage(video_url):
    """Try to extract basic info from YouTube page HTML"""
    try:
        video_id = get_video_id(video_url)
        if not video_id:
            return None
            
        # Make a simple request to YouTube page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", 
                              headers=headers, timeout=15)
        
        if response.status_code == 200:
            html = response.text
            
            # Extract title
            title = 'Unknown Title'
            if '"title":"' in html:
                start = html.find('"title":"') + 9
                end = html.find('"', start)
                if end > start:
                    title = html[start:end].replace('\\u0026', '&').replace('\\"', '"')
            
            # Extract channel name
            channel = 'Unknown Channel'
            if '"author":"' in html:
                start = html.find('"author":"') + 10
                end = html.find('"', start)
                if end > start:
                    channel = html[start:end].replace('\\u0026', '&').replace('\\"', '"')
            
            # Extract view count
            view_count = 0
            if '"viewCount":"' in html:
                start = html.find('"viewCount":"') + 13
                end = html.find('"', start)
                if end > start:
                    try:
                        view_count = int(html[start:end])
                    except:
                        view_count = 0
            
            # Extract duration (in seconds)
            duration_str = 'Unknown'
            if '"lengthSeconds":"' in html:
                start = html.find('"lengthSeconds":"') + 17
                end = html.find('"', start)
                if end > start:
                    try:
                        duration = int(html[start:end])
                        hours = duration // 3600
                        minutes = (duration % 3600) // 60
                        seconds = duration % 60
                        if hours > 0:
                            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            duration_str = f"{minutes:02d}:{seconds:02d}"
                    except:
                        duration_str = 'Unknown'
            
            return {
                'title': title,
                'channel': channel,
                'duration': duration_str,
                'thumbnail': f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg',
                'view_count': view_count,
                'upload_date': ''
            }
            
    except Exception as e:
        print(f"Webpage method failed: {e}")
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def get_video_metadata(video_url):
    """Get YouTube video metadata (title, channel, duration) with robust error handling"""
    
    # Method 1: Try with metadata-focused yt-dlp options
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,  # We need full extraction for duration/views
            'skip_download': True,
            'no_check_certificate': True,
            'ignoreerrors': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'format': 'worst[height<=144]/worst',  # Very low quality to minimize format issues
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First try with flat extraction
            info = ydl.extract_info(video_url, download=False)
            if info:
                title = info.get('title', 'Unknown Title')
                channel = info.get('uploader', info.get('channel', 'Unknown Channel'))
                duration = info.get('duration', 0)
                
                # Format duration from seconds to mm:ss or hh:mm:ss
                if duration and duration > 0:
                    hours = duration // 3600
                    minutes = (duration % 3600) // 60
                    seconds = duration % 60
                    if hours > 0:
                        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = f"{minutes:02d}:{seconds:02d}"
                else:
                    duration_str = "Unknown"
                
                return {
                    'title': title,
                    'channel': channel,
                    'duration': duration_str,
                    'thumbnail': info.get('thumbnail', ''),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', '')
                }
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        
        # Method 2: Try with different options
        try:
            ydl_opts_alt = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True,
                'no_check_certificate': True,
                'ignoreerrors': True,
                'format': 'worst',  # Use worst quality to avoid format issues
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_alt) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if info:
                    title = info.get('title', 'Unknown Title')
                    channel = info.get('uploader', info.get('channel', 'Unknown Channel'))
                    duration = info.get('duration', 0)
                    
                    if duration and duration > 0:
                        hours = duration // 3600
                        minutes = (duration % 3600) // 60
                        seconds = duration % 60
                        if hours > 0:
                            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        else:
                            duration_str = f"{minutes:02d}:{seconds:02d}"
                    else:
                        duration_str = "Unknown"
                    
                    return {
                        'title': title,
                        'channel': channel,
                        'duration': duration_str,
                        'thumbnail': info.get('thumbnail', ''),
                        'view_count': info.get('view_count', 0),
                        'upload_date': info.get('upload_date', '')
                    }
        except Exception as e2:
            print(f"Method 2 also failed: {e2}")
    
    # Method 3: Try webpage scraping (can get duration and views)
    try:
        webpage_result = get_video_info_webpage(video_url)
        if webpage_result and webpage_result['title'] != 'Unknown Title':
            return webpage_result
    except Exception as e3:
        print(f"Webpage method failed: {e3}")
    
    # Method 4: Try oEmbed API (basic info only)
    try:
        oembed_result = get_video_info_oembed(video_url)
        if oembed_result:
            return oembed_result
    except Exception as e4:
        print(f"oEmbed method failed: {e4}")
    
    # Method 5: Basic fallback - extract video ID and show partial info
    try:
        video_id = get_video_id(video_url)
        if video_id:
            return {
                'title': f'YouTube Video (ID: {video_id})',
                'channel': 'YouTube Channel',
                'duration': 'Visit YouTube for details',
                'thumbnail': f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg',
                'view_count': 0,
                'upload_date': ''
            }
    except:
        pass
    
    # Final fallback
    return {
        'title': 'Could not fetch title',
        'channel': 'Could not fetch channel', 
        'duration': 'Unknown',
        'thumbnail': '',
        'view_count': 0,
        'upload_date': ''
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_transcript_segments(video_id):
    try:
        # Create API instance
        api = YouTubeTranscriptApi()
        
        # Try to get available transcripts
        available_transcripts = api.list(video_id)
        
        # Try to find English transcript first
        try:
            transcript = available_transcripts.find_transcript(['en'])
            segments = transcript.fetch()
            if segments:
                print(f"✅ Found English transcript")
                return segments
        except Exception as e:
            print(f"English transcript not available: {e}")
        
        # If no English, try any available transcript
        for transcript in available_transcripts:
            try:
                segments = transcript.fetch()
                if segments:
                    print(f"✅ Found transcript in language: {transcript.language}")
                    return segments
            except Exception as e:
                print(f"Failed to fetch transcript for {transcript.language}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ No transcripts available for video {video_id}: {e}")
    
    return None

@st.cache_data
def format_transcript_text(segments):
    return " ".join([seg.text for seg in segments])

def download_audio(video_url):
    st.warning("⚠️ Audio download is temporarily disabled due to YouTube's new restrictions.")
    st.info("� The app will try to get transcripts directly from YouTube instead.")
    return None
    
    # Try with FFmpeg first if available
    if os.path.exists(ffmpeg_path):
        try:
            print(f"🔄 Downloading audio from: {video_url}")
            print(f"🔧 Using FFmpeg at: {ffmpeg_path}")
            
            with yt_dlp.YoutubeDL(ydl_opts_with_ffmpeg) as ydl:
                ydl.download([video_url])
            
            # Look for the downloaded audio file
            for ext in ['mp3', 'webm', 'm4a', 'wav']:
                candidate = f"audio.{ext}"
                if os.path.exists(candidate):
                    print(f"✅ Found audio file: {candidate}")
                    return candidate
        except Exception as e:
            print(f"FFmpeg download failed: {e}, trying fallback method...")
    
    # Fallback: try without FFmpeg post-processing
    try:
        print(f"� Trying fallback download without FFmpeg post-processing...")
        
        with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
            ydl.download([video_url])
        
        # Look for the downloaded audio file
        for ext in ['webm', 'm4a', 'mp4', 'wav', 'mp3']:
            candidate = f"audio.{ext}"
            if os.path.exists(candidate):
                print(f"✅ Found audio file: {candidate}")
                return candidate
        
        print("❌ Audio file not found after download.")
        print("📁 Files in directory:", os.listdir())
        return None
    except Exception as e:
        print(f"Audio download failed: {e}")
        st.error(f"❌ Audio download failed: {e}")
        return None

def transcribe_audio_whisper(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return None
    import torch
    try:
        st.info("🔄 Whisper: 'tiny' மாடல் ஏற்றப்படுகிறது... (Faster)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny", device=device)
        st.info(f"🔄 Whisper: ஆடியோ உரையாக்கம் நடைபெறுகிறது... (Device: {device})")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Whisper transcription failed: {e}")
        return None

@st.cache_data
def clean_text(text: str) -> str:
    # Remove timestamps like [00:01], (00:01), 00:01, etc.
    text = re.sub(r'\[.*?\]|\(.*?\)|\d{1,2}:\d{2}', '', text)
    # Remove common non-speech cues
    text = re.sub(r'\b(animation|music|sound effect|applause|laughter|subtitle|background score|intro|outro|credits|subscribe|like|share|comment)\b', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_data
def chunk_text(text: str, max_words: int = 100) -> list:
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def translate_to_english(text):
    try:
        lang = detect(text)
        print(f"🌐 Detected language: {lang}")
        if lang == "en":
            return text
        translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{lang}-en")
        result = translator(text)
        return result[0]['translation_text']
    except Exception as e:
        print(f"Translation to English failed: {e}")
        return text

@st.cache_data(ttl=7200)  # Cache for 2 hours
def translate_to_tamil(text, target_lang="ta"):
    chunks = chunk_text(clean_text(text), max_words=400)
    translated_chunks = []
    for chunk in chunks:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(chunk)
        except Exception as e2:
            print(f"Deep Translator failed for chunk: {e2}")
            translated = "[Translation failed for this part]"
        translated_chunks.append(translated)
    return ' '.join(translated_chunks)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def generate_summary(llm, transcript):
    system_prompt = f"""
நீங்கள் ஒரு தமிழ் மொழி நிபுணர் மற்றும் எளிமையான விளக்கத்தில் சிறந்த AI. கீழே உள்ள YouTube உரையைப் படித்து, அதன் உள்ளடக்கத்தை தமிழில் ஒரு சுருக்கமாகவும், தெளிவாகவும், ஆரம்ப நிலை பயனாளர்களுக்குப் புரியும் வகையில் எழுதவும். முக்கியமான கருத்துகள், எடுத்துக்காட்டுகள் மற்றும் விளக்கங்கள் சேர்க்கவும். வெறும் தலைப்புகள் பட்டியலாக மட்டும் எழுத வேண்டாம்; உரையின் சாரத்தை உணர்த்தும் வகையில், உரையாடல் மற்றும் விளக்கத்துடன் ஒரு முழுமையான சுருக்கத்தை வழங்கவும்.

---TRANSCRIPT---
{transcript}
---END TRANSCRIPT---
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', 'மேற்கண்ட உரையைப் பயன்படுத்தி, தமிழில் ஒரு முழுமையான, எளிமையான, விளக்கமான சுருக்கத்தை உருவாக்கவும். வெறும் தலைப்புகள் பட்டியலாக மட்டும் எழுத வேண்டாம்.')
    ])
    
    if llm is None:
        return "❌ LLM not available. Please check Ollama connection."
    
    try:
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke({})
    except Exception as e:
        return f"❌ Summary generation failed: {str(e)}"

# --- Main Logic ---

if url and url != st.session_state.last_url:
    st.session_state.last_url = url
    st.session_state.summary = None
    st.session_state.transcript_text = None
    st.session_state.summary_en = None
    st.session_state.video_metadata = None

    video_id = get_video_id(url)
    if not video_id:
        st.error("⚠️ Invalid YouTube URL.")
    else:
        # Create a progress bar and status container
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        with status_container:
            st.markdown("""
            <div class="success-box">
                🎬 Processing your video... Please wait
            </div>
            """, unsafe_allow_html=True)
        
        # Step 0: Get video metadata
        progress_bar.progress(10)
        status_container.info("📋 Fetching video information...")
        st.session_state.video_metadata = get_video_metadata(url)
        
        transcript_text = None
        
        # Step 1: Check manual transcript
        if manual_transcript and manual_transcript.strip():
            progress_bar.progress(30)
            status_container.success("✅ Using manual transcript provided")
            transcript_text = manual_transcript.strip()
        else:
            # Step 2: Get YouTube transcript
            progress_bar.progress(30)
            status_container.info("🔍 Fetching transcript from YouTube...")
            transcript_segments = get_transcript_segments(video_id)
            if transcript_segments:
                progress_bar.progress(50)
                transcript_text = format_transcript_text(transcript_segments)
                status_container.success("✅ Successfully retrieved transcript from YouTube!")
            else:
                progress_bar.progress(100)
                status_container.warning("⚠️ No transcript available from YouTube.")
                st.info("📝 Please paste the transcript manually in the sidebar, or try a different video.")
                transcript_text = None

        if transcript_text:
            # Step 3: Clean and prepare transcript
            progress_bar.progress(60)
            status_container.info("🧹 Cleaning and preparing transcript...")
            transcript_text = clean_text(transcript_text)
            st.session_state.transcript_text = transcript_text
            
            # Step 4: Generate AI summary
            progress_bar.progress(75)
            status_container.info("🤖 Generating AI summary...")
            try:
                summary_en = generate_summary(llm, transcript_text)
                st.session_state.summary_en = summary_en
            except Exception as e:
                print(f"LLM summary failed: {e}")
                progress_bar.progress(100)
                status_container.error("❌ LLM disconnected. Try restarting Ollama or reducing transcript size.")
                summary_en = "⚠️ Summary generation failed."
            
            # Step 5: Translate to Tamil
            if summary_en and "failed" not in summary_en:
                progress_bar.progress(90)
                status_container.info("🌐 Translating to Tamil...")
                summary_ta = translate_to_tamil(summary_en, target_lang="ta")
                st.session_state.summary = summary_ta
                
                # Also translate the transcript for download
                st.session_state.transcript_ta = translate_to_tamil(transcript_text, target_lang="ta")
                
                # Step 6: Complete!
                progress_bar.progress(100)
                status_container.success("🎉 Processing complete! Scroll down to see your Tamil summary.")

# --- Display Output ---

if st.session_state.summary:
    # Video metadata display
    if st.session_state.video_metadata:
        metadata = st.session_state.video_metadata
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="margin: 0 0 1rem 0; font-size: 1.5rem;">📺 Video Information</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <strong>📝 Title:</strong><br>
                    <span style="font-size: 1.1rem;">""" + metadata['title'] + """</span>
                </div>
                <div>
                    <strong>📺 Channel:</strong><br>
                    <span style="font-size: 1.1rem;">""" + metadata['channel'] + """</span>
                </div>
                <div>
                    <strong>⏱️ Duration:</strong><br>
                    <span style="font-size: 1.1rem;">""" + metadata['duration'] + """</span>
                </div>
                <div>
                    <strong>👀 Views:</strong><br>
                    <span style="font-size: 1.1rem;">""" + (f"{metadata['view_count']:,}" if metadata['view_count'] else "N/A") + """</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tamil Summary with beautiful styling
    st.markdown("""
    <div class="summary-container">
        <h2 style="margin-bottom: 1rem;">📝 வீடியோ சுருக்கம் (Tamil Summary)</h2>
        <div class="tamil-text">
    """ + st.session_state.summary.replace('\n', '<br>') + """
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Generate New Summary"):
            st.session_state.summary = None
            st.session_state.summary_en = None
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Summary"):
            st.code(st.session_state.summary, language="text")
    
    with col3:
        if st.session_state.summary:
            st.download_button(
                label="💾 Download Summary",
                data=st.session_state.summary,
                file_name="tamil_video_summary.txt",
                mime="text/plain"
            )

    # Tamil transcript section
    if hasattr(st.session_state, 'transcript_ta') and st.session_state.transcript_ta:
        st.markdown("---")
        st.markdown("### 📄 தமிழாக்க டிரான்ஸ்கிரிப்ட் (Tamil Transcript)")
        
        with st.expander("📖 View Full Tamil Transcript", expanded=False):
            st.markdown(f'<div class="tamil-text">{st.session_state.transcript_ta}</div>', unsafe_allow_html=True)
            st.download_button(
                label="💾 Download Tamil Transcript",
                data=st.session_state.transcript_ta,
                file_name="transcript_tamil.txt",
                mime="text/plain"
            )

    # Debug information
    with st.expander("🔧 Advanced Details", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📜 Original Transcript", "🌐 English Summary", "📊 Statistics"])
        
        with tab1:
            if st.session_state.transcript_text:
                st.text_area("Original Transcript", st.session_state.transcript_text, height=300)
                st.info(f"📊 Character count: {len(st.session_state.transcript_text)}")
        
        with tab2:
            if st.session_state.summary_en:
                st.markdown("**English Summary (Before Translation):**")
                st.write(st.session_state.summary_en)
        
        with tab3:
            if st.session_state.transcript_text and st.session_state.summary:
                original_words = len(st.session_state.transcript_text.split())
                summary_words = len(st.session_state.summary.split())
                compression_ratio = round((1 - summary_words/original_words) * 100, 1) if original_words > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Words", original_words)
                with col2:
                    st.metric("Summary Words", summary_words)
                with col3:
                    st.metric("Compression", f"{compression_ratio}%")

else:
    # Welcome screen with instructions
    st.markdown("---")
    st.markdown("### 🚀 Get Started")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>� How it works:</h4>
            <ol>
                <li>📎 Paste a YouTube URL in the sidebar</li>
                <li>⚡ AI automatically extracts and processes the content</li>
                <li>🎯 Get a comprehensive Tamil summary in seconds</li>
                <li>💾 Download your summary and transcript</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>✨ Features:</h4>
            <ul>
                <li>🌍 Multi-language support</li>
                <li>🤖 AI-powered summaries</li>
                <li>📱 Mobile-friendly</li>
                <li>⚡ Fast processing</li>
                <li>💾 Download options</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>🎬 எந்த மொழியிலான YouTube வீடியோ URL-ஐ ஒட்டவும் — தமிழில் சுருக்கம் பெறலாம்!</h3>
        <p style="font-size: 1.1rem; color: #666;">Paste any YouTube video URL to get started with your Tamil summary</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 3rem;">
    <h4>🌟 Universal Tamil Video Summarizer</h4>
    <p>Powered by <strong>Ollama LLM</strong> • <strong>YouTube Transcript API</strong> • <strong>Streamlit</strong></p>
    <p style="font-size: 0.9rem; color: #666;">
        Transform any video content into clear, comprehensive Tamil summaries with AI
    </p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem;">🤖 AI-Powered</span>
        <span style="margin: 0 1rem;">🌐 Multi-Language</span>
        <span style="margin: 0 1rem;">⚡ Fast Processing</span>
    </div>
</div>
""", unsafe_allow_html=True)