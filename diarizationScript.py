import argparse
from zoomDiarization import ZoomSpeakerDiarization
import os

def main():
    """Main function to run the diarization"""   
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--video_dir', default='/work/users/s/m/smerrill/Albemarle', help='Path to directory containing zoom videos')
    parser.add_argument('--save_dir', default='/work/users/s/m/smerrill/Albemarle/transcripts', help='Path to directory to save transcripts')
    parser.add_argument('--sampling_rate', '-s', type=float, default=1.0,
                        help='Frames per second to sample (default: 1)')
    
    args = parser.parse_args()
    

    video_files = os.listdir(args.video_dir)

    for video_file in video_files:
        try:
            save_name = os.path.join(args.save_dir, video_file.split('.')[0] + '.txt')
            video_path = os.path.join(args.video_dir, video_file)
            diarizer = ZoomSpeakerDiarization(video_path)
            diarized_transcript = diarizer.process_video(sampling_rate=args.sampling_rate)
            diarizer.save_transcript(diarized_transcript, save_name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()

