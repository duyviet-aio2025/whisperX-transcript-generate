"""
Video creation module using ffmpeg-python for generating videos with synchronized subtitles
"""

import ffmpeg
import os
import tempfile
import json
from typing import List, Dict, Any, Tuple, Optional
import subprocess
import shutil


class VideoCreator:
    """Create videos with synchronized subtitles using ffmpeg-python"""
    
    def __init__(self):
        self.temp_files = []
        self.default_style = {
            'fontsize': 48,
            'fontcolor': 'white',
            'fontfile': None,  # Use system default
            'box': 1,
            'boxcolor': 'black@0.8',
            'boxborderw': 5,
            'x': '(w-text_w)/2',  # Center horizontally
            'y': '(h-text_h)/2'  # Center vertically - chính giữa video
        }
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try alternative paths for cloud deployments
            for ffmpeg_path in ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', 'ffmpeg']:
                try:
                    subprocess.run([ffmpeg_path, '-version'], capture_output=True, check=True)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            return False
    
    def create_srt_file(self, segments: List[Dict], output_path: str, include_speakers: bool = True) -> str:
        """Create SRT subtitle file from transcript segments"""
        srt_content = ""
        
        for i, segment in enumerate(segments, 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', start_time + 1)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'Unknown')
            
            # Format timestamps for SRT (HH:MM:SS,mmm)
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # Format text with optional speaker label
            if include_speakers and speaker != 'Unknown':
                formatted_text = f"[{speaker}] {text}"
            else:
                formatted_text = text
            
            srt_content += f"{i}\n{start_srt} --> {end_srt}\n{formatted_text}\n\n"
        
        # Write SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        self.temp_files.append(output_path)
        return output_path
    
    def create_ass_file(self, segments: List[Dict], output_path: str, 
                       style_config: Optional[Dict] = None, include_speakers: bool = True) -> str:
        """Create ASS subtitle file with advanced styling"""
        if style_config is None:
            style_config = self.default_style.copy()
        
        # ASS file header
        ass_content = """[Script Info]
Title: WhisperX Generated Subtitles
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{fontsize},&Hffffff,&Hffffff,&H0,&H80000000,0,0,0,0,100,100,0,0,1,2,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(fontsize=style_config.get('fontsize', 48))
        
        # Add subtitle events
        for segment in segments:
            start_time = segment.get('start', 0)
            end_time = segment.get('end', start_time + 1)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'Unknown')
            
            # Format timestamps for ASS (H:MM:SS.cc)
            start_ass = self._seconds_to_ass_time(start_time)
            end_ass = self._seconds_to_ass_time(end_time)
            
            # Format text with optional speaker label
            if include_speakers and speaker != 'Unknown':
                formatted_text = f"[{speaker}] {text}"
            else:
                formatted_text = text
            
            # Escape special characters
            formatted_text = formatted_text.replace('\n', '\\N')
            
            ass_content += f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{formatted_text}\n"
        
        # Write ASS file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        self.temp_files.append(output_path)
        return output_path
    
    def create_video_with_subtitles(self, 
                                  audio_path: str,
                                  segments: List[Dict],
                                  output_path: str,
                                  resolution: Tuple[int, int] = (1280, 720),
                                  background_color: str = '#1a1a2e',
                                  subtitle_style: Optional[Dict] = None,
                                  include_speakers: bool = True) -> str:
        """
        Create video with audio and synchronized subtitles using ASS format
        
        Args:
            audio_path: Path to input audio file
            segments: List of transcript segments with timing
            output_path: Path for output video file
            resolution: Video resolution as (width, height)
            background_color: Background color (hex or color name)
            subtitle_style: Dictionary with subtitle styling options
            include_speakers: Whether to include speaker labels
        
        Returns:
            Path to created video file
        """
        if not self.check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg first.")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not segments:
            raise ValueError("No transcript segments provided")
        
        # Validate and fix timing first
        validation_result = self.validate_segments(segments)
        validated_segments = validation_result["segments"]
        
        # Merge default style with provided style
        style = self.default_style.copy()
        if subtitle_style:
            style.update(subtitle_style)
        
        try:
            # Create ASS subtitle file
            subtitle_path = os.path.splitext(output_path)[0] + '.ass'
            self.create_ass_file(validated_segments, subtitle_path, style, include_speakers)
            
            # Get audio duration
            audio_info = ffmpeg.probe(audio_path)
            duration = float(audio_info['streams'][0]['duration'])
            
            # Create video pipeline using ffmpeg-python
            width, height = resolution
            
            # Create blank video with background color
            video_input = (
                ffmpeg
                .input('color=c={}:s={}x{}:d={}'.format(
                    background_color, width, height, duration
                ), f='lavfi')
            )
            
            # Load audio
            audio_input = ffmpeg.input(audio_path)
            
            # Apply ASS subtitles
            video_with_subs = video_input.filter('subtitles', subtitle_path)
            
            # Combine video and audio
            output = ffmpeg.output(
                video_with_subs,
                audio_input,
                output_path,
                vcodec='libx264',
                acodec='aac',
                preset='medium',
                crf=23,
                r=25  # 25 fps
            )
            
            # Run ffmpeg
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            raise RuntimeError(f"FFmpeg error: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Video creation failed: {str(e)}")
    


    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    
    def _color_to_hex(self, color: str) -> str:
        """Convert color name to hex for ASS format"""
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'magenta': 'FF00FF',
            'orange': 'FFA500',
            'pink': 'FFC0CB',
            'lime': '00FF00'
        }
        return color_map.get(color.lower(), 'FFFFFF')
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'size': int(probe['format']['size']),
                'format': probe['format']['format_name']
            }
            
            if video_stream:
                info.update({
                    'width': int(video_stream['width']),
                    'height': int(video_stream['height']),
                    'fps': eval(video_stream['r_frame_rate']),
                    'video_codec': video_stream['codec_name']
                })
            
            if audio_stream:
                info.update({
                    'audio_codec': audio_stream['codec_name'],
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': int(audio_stream['channels'])
                })
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to get video info: {str(e)}")
    
    def validate_segments(self, segments: List[Dict]) -> Dict[str, Any]:
        """Validate and fix timing issues in segments"""
        if not segments:
            return {
                'segments': [],
                'issues': ['No segments provided'],
                'fixed_count': 0,
                'original_count': 0
            }
        
        fixed_segments = []
        issues = []
        
        for i, segment in enumerate(segments):
            start = segment.get('start', 0)
            end = segment.get('end', start + 1)
            text = segment.get('text', '').strip()
            
            # Fix timing issues
            if start < 0:
                issues.append(f"Segment {i}: Negative start time {start} fixed to 0")
                start = 0
            
            if end <= start:
                issues.append(f"Segment {i}: End time {end} <= start time {start}, fixed")
                end = start + 1
            
            if not text:
                issues.append(f"Segment {i}: Empty text, skipping")
                continue
            
            # Check for overlapping with previous segment
            if fixed_segments and start < fixed_segments[-1]['end']:
                issues.append(f"Segment {i}: Overlap with previous segment, adjusting")
                start = fixed_segments[-1]['end']
                if end <= start:
                    end = start + 1
            
            fixed_segments.append({
                'start': start,
                'end': end,
                'text': text,
                'speaker': segment.get('speaker', 'Unknown')
            })
        
        return {
            'segments': fixed_segments,
            'issues': issues,
            'fixed_count': len(fixed_segments),
            'original_count': len(segments)
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        self.temp_files.clear()
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup_temp_files()


# Convenience functions
def create_video_from_transcript(audio_path: str, 
                                segments: List[Dict],
                                output_path: str,
                                **kwargs) -> str:
    """
    Convenience function to create video from transcript
    
    Args:
        audio_path: Path to input audio file
        segments: List of transcript segments
        output_path: Path for output video
        **kwargs: Additional options for video creation
    
    Returns:
        Path to created video file
    """
    creator = VideoCreator()
    try:
        return creator.create_video_with_subtitles(
            audio_path=audio_path,
            segments=segments,
            output_path=output_path,
            **kwargs
        )
    finally:
        creator.cleanup_temp_files()



