import os
import sys

cudnn_path = "/work/users/s/m/smerrill/.conda/envs/llm/lib/python3.7/site-packages/nvidia/cudnn/lib"
os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["PATH"] = "/work/users/s/m/smerrill/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]

import numpy as np
import cv2
from moviepy import VideoFileClip

import whisperx
from pyannote.audio import Pipeline
from pyannote.core import Segment
import argparse


def main():
    """Main function to run the diarization"""   
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--video_file', help='Path to zoom video')
    parser.add_argument('--save_dir', default='/work/users/s/m/smerrill/Albemarle/whisperDiarization', help='Path to directory to save whisperDiarization')
    
    args = parser.parse_args()
    video_file_name = args.video_file
    audio_file_name = args.video_file.replace('.mp4', '.wav')
    vid_id = video_file_name.split('/')[-1].split('.')[0]

    print(f"Attempting to Diarize {video_file_name}")

    print(f"Creating Audio File {audio_file_name}")
    #video_clip = VideoFileClip(video_file_name)
    #video_clip.audio.write_audiofile(audio_file_name)


    model = whisperx.load_model("large-v2", device="cuda", download_root='/work/users/s/m/smerrill/LocalView')
    result = model.transcribe(audio_file_name)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file_name, "cuda")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="removed")
    diarization = pipeline(audio_file_name, min_speakers=None, max_speakers=None)

    # Add speaker labels to WhisperX segments
    result_segments = result_aligned["segments"]  # or result["segments"] if you skip alignment

    for segment in result_segments:
        segment_start = segment["start"]
        segment_end = segment["end"]

        # Find the speaker active during this segment
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= segment_start <= turn.end or turn.start <= segment_end <= turn.end:
                segment["speaker"] = speaker
                break
            else:
                segment["speaker"] = "unknown"

    np.save(os.path.join(args.save_dir, f'{vid_id}.npy'), result_segments)


if __name__ == "__main__":
    main()

