import cv2
import numpy as np
import whisper
import os
import tempfile
from moviepy import VideoFileClip
import re
import time
import easyocr

os.environ["PATH"] = "/work/users/s/m/smerrill/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]

class ZoomSpeakerDiarization:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.speaker_changes = []
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.whisper_model = whisper.load_model("large-v2", 
                                                 device="cuda",
                                                 download_root='/work/users/s/m/smerrill/models')


    def ocr_name_from_box(self, box_img, debug=True, paragraph=True):
        """
        Uses EasyOCR to extract text from a cropped participant name box.
        Returns the best detected text string.
        """
        # Run OCR
        results = self.reader.readtext(box_img, detail=0, paragraph=paragraph)  # detail=0 returns just text[2][3]

        # Combine results into one string
        if isinstance(results, list):
            text = ' '.join(results).strip()
        else:
            text = str(results).strip()

        if debug:
            print("EasyOCR Output:", repr(text))
            plt.figure(figsize=(6,2))
            plt.imshow(box_img)
            plt.title(f"OCR: {text}")
            plt.axis('off')
            plt.show()

        return text if text else "No Speaker"

    def find_speaker(self, frame, overlay=True, debug=True):
        """
        Attempts to find the active speaker region in a Zoom screenshot.
        Handles:
          - Green/yellow border (participant grid)
          - Fixed overlay in upper right (screen share mode)
          - Yellow pop-up (Zoom's speaker label)
        Returns:
          frame_with_box, (x, y, w, h) of detected box, cropped box image
        """
        frame_disp = frame.copy()
        h, w = frame.shape[:2]
        detected_box = None
        cutout = None

        # 1. Try to find green/yellow border (participant grid)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([20, 40, 60])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_rect = None
        max_area = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, ww, hh = cv2.boundingRect(approx)
                area = ww * hh
                if area > max_area and ww > 40 and hh > 40:
                    max_rect = (x, y, ww, hh)
                    max_area = area
        if max_rect:
            detected_box = max_rect
            if overlay:
                cv2.rectangle(frame_disp, (detected_box[0], detected_box[1]),
                              (detected_box[0] + detected_box[2], detected_box[1] + detected_box[3]),
                              (0, 255, 0), 3)
            cutout = frame[detected_box[1]:detected_box[1]+detected_box[3], detected_box[0]:detected_box[0]+detected_box[2]]
            if debug:
                print("Detected green/yellow border (participant grid).")
        else:
            # Fallback: Fixed overlay in upper right (screen share mode)
            # Use proportions to handle resizing
            overlay_w = int(w * 0.18)  # ~26% of width (tune as needed)
            overlay_h = int(h * 0.18)  # ~20% of height (tune as needed)
            x1 = w - overlay_w
            y1 = 0
            detected_box = (x1, y1, overlay_w, overlay_h)
            cutout = frame[y1:y1+overlay_h, x1:x1+overlay_w]
            if overlay:
                cv2.rectangle(frame_disp, (x1, y1), (x1+overlay_w, y1+overlay_h), (255, 0, 0), 3)
            if debug:
                print("Detected fixed overlay (upper right).")

        if debug:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB))
            plt.title("Detected Speaker Box")
            plt.axis('off')
            plt.show()
            if cutout is not None:
                plt.figure(figsize=(3, 2))
                plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
                plt.title("Speaker Crop")
                plt.axis('off')
                plt.show()

        return frame_disp, detected_box, cutout
    

    def determine_active_speaker(self, frame):
        result_frame, speaker_box, cutout = self.find_speaker(frame, overlay=True, debug=False)
        name = self.ocr_name_from_box(self.crop_name_only(cutout), debug=False)
        return name

    
    def extract_audio(self):
        """Extract audio from video for transcription"""
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        video_clip = VideoFileClip(self.video_path)
        video_clip.audio.write_audiofile(temp_audio)
        video_clip.close()
        return temp_audio
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        
        return result
    
    def process_video(self, sampling_rate=1):
        """Process the video to create speaker diarization"""
        print(f"Processing video: {self.video_path}")
        print(f"Total frames: {self.total_frames}, FPS: {self.fps}")
        
        # Extract and transcribe audio
        audio_path = self.extract_audio()
        transcript_data = self.transcribe_audio(audio_path)
        
        # Process video frames at the specified sampling rate
        frame_interval = int(self.fps / sampling_rate)
        current_frame = 0
        last_speaker = None
        
        while current_frame < self.total_frames:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.capture.read()
            
            if not ret:
                break
                
            timestamp = current_frame / self.fps
            active_speaker = self.determine_active_speaker(frame)
            
            # Record speaker change
            if active_speaker != last_speaker:
                self.speaker_changes.append((timestamp, active_speaker))
                last_speaker = active_speaker
            
            # Output progress
            if current_frame % (frame_interval * 10) == 0:
                progress = (current_frame / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% - Current speaker: {active_speaker}")
            
            current_frame += frame_interval
        
        # Clean up
        self.capture.release()
        os.unlink(audio_path)
        
        # Map speaker changes to transcript segments
        return self.map_speakers_to_transcript(transcript_data)
    
    def map_speakers_to_transcript(self, transcript_data):
        """Map detected speakers to transcript segments"""
        segments = transcript_data["segments"]
        diarized_transcript = []
        
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            # Find the speaker active at the start of this segment
            speaker = "Unknown"
            for timestamp, spk in reversed(self.speaker_changes):
                if timestamp <= start_time:
                    speaker = spk
                    break
            
            diarized_transcript.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": text
            })
        
        return diarized_transcript
    
    
    def crop_name_only(self, box_img, height_ratio=0.22, width_ratio=0.6, bottom_padding=4):
        h, w = box_img.shape[:2]
        name_h = int(h * height_ratio)
        name_w = int(w * width_ratio)
        # Remove bottom padding to avoid colored lines
        name_region = box_img[h - name_h:h - bottom_padding, 0:name_w]
        return name_region
    def clean_crop_above_border(self, img, border_px=5):
        """Crop a few pixels above the bottom to remove colored border."""
        h, w = img.shape[:2]
        return img[:h-border_px, :]
    
    
    def save_transcript(self, diarized_transcript, output_file):
        """Save the diarized transcript to a file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in diarized_transcript:
                start_time = self.format_timestamp(segment["start"])
                end_time = self.format_timestamp(segment["end"])
                f.write(f"[{start_time} --> {end_time}] {segment['speaker']}: {segment['text']}\n")
        
        print(f"Transcript saved to {output_file}")
    
    def format_timestamp(self, seconds):
        """Format seconds into HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    