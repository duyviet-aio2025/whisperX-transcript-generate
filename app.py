import streamlit as st
import whisperx
import gc
import torch
import tempfile
import os
import pandas as pd
from typing import Optional, Dict, Any
import json
from dotenv import load_dotenv
from video_creator import VideoCreator

load_dotenv() 

# Configure page
st.set_page_config(
    page_title="WhisperX Transcript Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .transcript-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .speaker-label {
        font-weight: bold;
        color: #2e86de;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class WhisperXTranscriber:
    def __init__(self):
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_transcription_model(self, model_name: str = "large-v2", compute_type: str = "float32"):
        """Load WhisperX transcription model"""
        try:
            if self.model is not None:
                del self.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.model = whisperx.load_model(model_name, self.device, compute_type=compute_type)
            return True
        except Exception as e:
            st.error(f"Error loading transcription model: {str(e)}")
            return False
    
    def load_alignment_model(self, language_code: str):
        """Load alignment model for the specified language"""
        try:
            if self.align_model is not None:
                del self.align_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code, 
                device=self.device
            )
            return True
        except Exception as e:
            st.error(f"Error loading alignment model: {str(e)}")
            return False
    
    def load_diarization_model(self, hf_token: str):
        """Load speaker diarization model"""
        try:
            if self.diarize_model is not None:
                del self.diarize_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token, 
                device=self.device
            )
            return True
        except Exception as e:
            st.error(f"Error loading diarization model: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_file_path: str, language: Optional[str] = None, 
                        batch_size: int = 16, enable_diarization: bool = True, 
                        hf_token: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio file with optional speaker diarization"""
        try:
            # Load audio
            audio = whisperx.load_audio(audio_file_path)
            
            # Transcribe
            if language and language != "auto":
                result = self.model.transcribe(audio, batch_size=batch_size, language=language)
            else:
                result = self.model.transcribe(audio, batch_size=batch_size)
            
            detected_language = result.get("language", "unknown")
            
            # Align transcription
            if self.load_alignment_model(detected_language):
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.align_metadata, 
                    audio, 
                    self.device, 
                    return_char_alignments=False
                )
            
            # Speaker diarization
            if enable_diarization and hf_token:
                if self.load_diarization_model(hf_token):
                    diarize_segments = self.diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
            
            return {
                "success": True,
                "language": detected_language,
                "segments": result.get("segments", []),
                "full_text": self._extract_full_text(result.get("segments", []))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_full_text(self, segments):
        """Extract full text from segments"""
        return " ".join([segment.get("text", "").strip() for segment in segments])

def format_time(seconds):
    """Format seconds to MM:SS format"""
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def display_transcript(segments):
    """Display formatted transcript with timestamps and speakers"""
    if not segments:
        st.warning("No transcript segments found.")
        return
    
    st.markdown("### 📝 Transcript")
    
    # Create DataFrame for better display
    transcript_data = []
    for i, segment in enumerate(segments):
        start_time = format_time(segment.get("start"))
        end_time = format_time(segment.get("end"))
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "Unknown")
        
        transcript_data.append({
            "Time": f"{start_time} - {end_time}",
            "Speaker": speaker,
            "Text": text
        })
    
    # Display as dataframe
    df = pd.DataFrame(transcript_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Also display as formatted text
    st.markdown("### 📋 Formatted Text")
    formatted_text = ""
    
    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()
        start_time = format_time(segment.get("start"))
        
        if formatted_text:
            formatted_text += "\n\n"
        formatted_text += f"**[{start_time}] {speaker}:** {text}"
    
    st.markdown(f'<div class="transcript-container">{formatted_text}</div>', 
                unsafe_allow_html=True)

def show_video_creation_section(segments, uploaded_file, temp_audio_path):
    """Display video creation section with controls"""
    st.markdown("---")
    st.markdown("### 🎬 Create Video with Subtitles")
    
    with st.expander("🎥 Video Creation Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📐 Video Settings")
            resolution_options = {
                "1920x540": (1920, 540),
                "1280x360": (1280, 360), 
                "854x240": (854, 240),
                "640x180": (640, 180)
            }
            
            selected_resolution = st.selectbox(
                "Resolution",
                list(resolution_options.keys()),
                index=1,
                help="Higher resolution = better quality but larger file size"
            )
            resolution = resolution_options[selected_resolution]
            
            # Background color picker
            st.markdown("**Background Color:**")
            bg_preset = st.selectbox(
                "Color Preset",
                ["Dark Blue", "Black", "Dark Gray", "Navy", "Dark Purple", "Custom"],
                index=0
            )
            
            if bg_preset == "Custom":
                bg_color = st.color_picker("Choose Background Color", "#1a1a2e")
            else:
                color_presets = {
                    "Dark Blue": "#1a1a2e",
                    "Black": "#000000",
                    "Dark Gray": "#282828",
                    "Navy": "#191932",
                    "Dark Purple": "#1e1428"
                }
                bg_color = color_presets[bg_preset]
        
        with col2:
            st.markdown("#### 📝 Subtitle Style")
            font_size = st.slider("Font Size", 16, 72, 48)
            
            font_color = st.selectbox(
                "Text Color",
                ["white", "yellow", "cyan", "lime", "orange", "pink"],
                index=0
            )
            
            show_speaker_labels = st.checkbox(
                "Show Speaker Labels",
                value=True,
                help="Display [SPEAKER_XX] labels in subtitles"
            )
    
    # Create video button
    if st.button("🎯 Create Video with Subtitles", type="primary", use_container_width=True):
        try:
            with st.spinner("🎬 Creating video with subtitles..."):
                # Initialize video creator
                video_creator = VideoCreator()
                
                # Check if FFmpeg is available
                if not video_creator.check_ffmpeg():
                    st.error("❌ FFmpeg not found! Please install FFmpeg first.")
                    with st.expander("🔧 FFmpeg Installation Guide"):
                        st.markdown("""
                        **Install FFmpeg:**
                        
                        **macOS:**
                        ```bash
                        brew install ffmpeg
                        ```
                        
                        **Ubuntu/Debian:**
                        ```bash
                        sudo apt update && sudo apt install ffmpeg
                        ```
                        
                        **Windows:**
                        1. Download from https://ffmpeg.org/download.html
                        2. Extract and add to PATH
                        3. Or use winget: `winget install ffmpeg`
                        """)
                    return
                
                # Progress tracking setup
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create output filename
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_video_path = f"{base_name}_with_subtitles.mp4"
                
                status_text.text("🎬 Creating video with subtitles...")
                progress_bar.progress(50)
                
                # Prepare subtitle style
                subtitle_style = {
                    'fontsize': font_size,
                    'fontcolor': font_color,
                    'box': 1,
                    'boxcolor': 'black@0.8',
                    'boxborderw': 5
                }
                
                # Create video with ASS subtitles
                video_path = video_creator.create_video_with_subtitles(
                    audio_path=temp_audio_path,
                    segments=segments,
                    output_path=output_video_path,
                    resolution=resolution,
                    background_color=bg_color,
                    subtitle_style=subtitle_style,
                    include_speakers=show_speaker_labels
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Video creation completed!")
                
                # Success message
                st.success("🎉 Video with subtitles created successfully!")
                
                # Video preview and download
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎥 Video Preview")
                    if os.path.exists(video_path):
                        # Display video
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                        
                        # File info
                        try:
                            video_info = video_creator.get_video_info(video_path)
                            file_size = video_info.get('size', 0) / (1024 * 1024)  # MB
                            duration = video_info.get('duration', 0)
                            st.info(f"📊 File size: {file_size:.1f} MB | Duration: {duration:.1f}s")
                        except:
                            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                            st.info(f"📊 File size: {file_size:.1f} MB")
                
                with col2:
                    st.markdown("#### 💾 Download Options")
                    
                    # Download video
                    with open(video_path, 'rb') as video_file:
                        st.download_button(
                            "📥 Download Video (MP4)",
                            video_file.read(),
                            file_name=f"{base_name}_with_subtitles.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    
                    # Download ASS file
                    ass_path = f"{base_name}_subtitles.ass"
                    if os.path.exists(ass_path):
                        with open(ass_path, 'r', encoding='utf-8') as ass_file:
                            st.download_button(
                                "📄 Download Subtitles (ASS)",
                                ass_file.read(),
                                file_name=f"{base_name}_subtitles.ass",
                                mime="text/plain",
                                use_container_width=True
                            )
                    
                    # Video specifications
                    st.markdown("**Video Specs:**")
                    st.text(f"Resolution: {selected_resolution}")
                    st.text(f"Format: MP4 (H.264)")
                    st.text(f"Subtitles: ASS embedded")
                    # Get duration from segments
                    if segments:
                        duration = segments[-1].get('end', 0)
                        st.text(f"Duration: {duration:.1f}s")
                        st.text(f"Segments: {len(segments)}")
                
                # Clean up
                video_creator.cleanup_temp_files()
        
        except Exception as e:
            st.error(f"❌ Error creating video: {str(e)}")
            
            # Show debug info
            if st.checkbox("Show debug info"):
                st.text(f"Error details: {str(e)}")
                st.text(f"Audio path exists: {os.path.exists(temp_audio_path) if temp_audio_path else 'No audio path'}")
                st.text(f"Segments count: {len(segments) if segments else 0}")
                
                # Check FFmpeg
                try:
                    import subprocess
                    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.text("✅ FFmpeg available")
                    else:
                        st.text("❌ FFmpeg not working")
                except FileNotFoundError:
                    st.text("❌ FFmpeg not found")

def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ WhisperX Transcript Generator</h1>', 
                unsafe_allow_html=True)
    
    # Initialize transcriber and session state
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = WhisperXTranscriber()
    
    # Initialize session state for transcript results
    if 'transcript_result' not in st.session_state:
        st.session_state.transcript_result = None
    
    # Initialize session state for uploaded file info
    if 'uploaded_file_info' not in st.session_state:
        st.session_state.uploaded_file_info = None
    
    # Initialize session state for temporary audio path
    if 'temp_audio_path' not in st.session_state:
        st.session_state.temp_audio_path = None
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # HuggingFace Token
    hf_token = st.sidebar.text_input(
        "🔑 HuggingFace Token", 
        type="password",
        help="Required for speaker diarization. Get your token from https://huggingface.co/settings/tokens",
        value=os.getenv("TOKEN", "")
    )
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "🤖 Model",
        ["large-v3", "large-v2", "medium", "small", "base", "tiny"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    # Language selection
    languages = {
        "auto": "Auto-detect",
        "vi": "Vietnamese",
        "en": "English",
    }
    
    selected_language = st.sidebar.selectbox(
        "🌐 Language",
        list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0
    )
    
    # Advanced settings
    st.sidebar.header("🔧 Advanced Settings")
    
    compute_type = st.sidebar.selectbox(
        "💾 Compute Type",
        ["float32", "float16", "int8"],
        index=0,
        help="float16/int8 use less GPU memory but may reduce accuracy"
    )
    
    batch_size = st.sidebar.slider(
        "📦 Batch Size",
        min_value=1,
        max_value=32,
        value=16,
        help="Reduce if you have low GPU memory"
    )
    
    enable_diarization = st.sidebar.checkbox(
        "👥 Enable Speaker Diarization",
        value=True,
        help="Identify different speakers (requires HuggingFace token)"
    )
    
    # Main content
    st.markdown("### 📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file or same file
        current_file_info = {
            'name': uploaded_file.name,
            'size': uploaded_file.size
        }
        
        # Reset transcript result if new file is uploaded
        if (st.session_state.uploaded_file_info is None or 
            st.session_state.uploaded_file_info != current_file_info):
            st.session_state.transcript_result = None
            st.session_state.uploaded_file_info = current_file_info
            # Clean up old temp file
            if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
                os.unlink(st.session_state.temp_audio_path)
                st.session_state.temp_audio_path = None
        
        if st.button("🔄 New Transcription", type="secondary"):
            # Clear session state
            st.session_state.transcript_result = None
            if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
                os.unlink(st.session_state.temp_audio_path)
            st.session_state.temp_audio_path = None
            st.rerun()

        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Audio player
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')
        
        # Show existing transcript if available
        if st.session_state.transcript_result is not None:
            result = st.session_state.transcript_result
            st.success(f"🎉 Transcription completed! Detected language: {result['language']}")
            
            # Display results
            if result["segments"]:
                display_transcript(result["segments"])
                
                # Video creation section
                show_video_creation_section(result["segments"], uploaded_file, st.session_state.temp_audio_path)
                
                # Download options
                st.markdown("### 💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # JSON download
                    json_data = json.dumps(result["segments"], indent=2, ensure_ascii=False)
                    st.download_button(
                        "📄 Download JSON",
                        json_data,
                        file_name=f"{uploaded_file.name}_transcript.json",
                        mime="application/json"
                    )
                
                with col2:
                    # Text download
                    st.download_button(
                        "📝 Download Text",
                        result["full_text"],
                        file_name=f"{uploaded_file.name}_transcript.txt",
                        mime="text/plain"
                    )
                
                with col3:
                    # CSV download
                    df = pd.DataFrame([{
                        "start_time": segment.get("start"),
                        "end_time": segment.get("end"),
                        "speaker": segment.get("speaker", "Unknown"),
                        "text": segment.get("text", "").strip()
                    } for segment in result["segments"]])
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        csv_data,
                        file_name=f"{uploaded_file.name}_transcript.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ No transcript generated. Please check your audio file.")
        
        # Transcribe button (only show if no transcript exists)
        elif st.button("🎯 Generate Transcript", type="primary", use_container_width=True):
            # Validation
            if enable_diarization and not hf_token:
                st.error("🔑 HuggingFace token is required for speaker diarization!")
                return
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            try:
                with st.spinner("🔄 Loading models and processing audio..."):
                    # Load transcription model
                    if not st.session_state.transcriber.load_transcription_model(model_name, compute_type):
                        return
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🎵 Transcribing audio...")
                    progress_bar.progress(25)
                    
                    # Transcribe
                    result = st.session_state.transcriber.transcribe_audio(
                        temp_file_path,
                        language=selected_language if selected_language != "auto" else None,
                        batch_size=batch_size,
                        enable_diarization=enable_diarization,
                        hf_token=hf_token if enable_diarization else None
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    if result["success"]:
                        # Save result to session state
                        st.session_state.transcript_result = result
                        st.session_state.temp_audio_path = temp_file_path
                        
                        # Rerun to show the results
                        st.rerun()
                    else:
                        st.error(f"❌ Transcription failed: {result['error']}")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
            
            finally:
                # Clean up temp file only if transcription failed
                if 'result' in locals() and not result.get("success", False):
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

if __name__ == "__main__":
    main()
