import argparse
from zoomDiarization import ZoomSpeakerDiarization
import os

def main():
    """Main function to run the diarization"""   
    parser = argparse.ArgumentParser(description='Zoom Meeting Speaker Diarization')
    parser.add_argument('--video_file', help='Path to zoom video')
    parser.add_argument('--save_dir', default='/work/users/s/m/smerrill/Albemarle/speakerChanges', help='Path to directory to save transcripts')
    parser.add_argument('--sampling_rate', '-s', type=float, default=2.0,
                        help='Frames per second to sample (default: 1)')
    
    args = parser.parse_args()    
    print(f"Attempting to Download {args.video_file}")
    try:
        save_name = os.path.join(args.save_dir, args.video_file.split('/')[-1].replace('.mp4', '.txt'))
        print(f"Save Name: {save_name}")
        
        diarizer = ZoomSpeakerDiarization(args.video_file)
        diarized_transcript = diarizer.process_video(sampling_rate=args.sampling_rate)
        # no longer using this basic logic to construct transcripts
        # diarizer.save_transcript(diarized_transcript, save_name)

        # instead just save the video mappings
        save_name = os.path.join(args.save_dir, args.video_file.split('/')[-1].replace('.mp4', '.npy'))
        np.save(save_name, diarizer.speaker_changes)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()

