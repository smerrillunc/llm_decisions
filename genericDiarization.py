import numpy as np
import pickle
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.data.path.append("/work/users/s/m/smerrill/nltk")
nltk.download('vader_lexicon')

from pyannote.audio import Pipeline
from pyannote.core import Segment
import os
import argparse
import whisperx

# ffmpeg
os.environ["PATH"] = "/work/users/s/m/smerrill/ffmpeg-7.0.2-amd64-static:" + os.environ["PATH"]

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Read file content.')
	parser.add_argument("-mins", "--min_speakers", type=int, default=4, help='Min Speakers')
	parser.add_argument("-maxs", "--max_speakers", type=int, default=12, help='Max Speakers')
	parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/LocalView', help='Path to YoutubeID file.  This will also be where output featuers are saved')
	args = vars(parser.parse_args())

	os.makedirs(args['save_path'] + '/transcriptions', exist_ok=True)

	audio_path = args['save_path'] + '/audio'
	audio_files = [x for x in os.listdir(audio_path) if '.mp3' in x]

	for audio_file in audio_files:
		print(f"Processing {audio_file}")
		try:
			save_name = audio_file.replace('mp3', 'npy')
			save_path = os.path.join(args['save_path'], 'transcriptions', save_name)
			hf_token = "REMOVED"

			#model = whisperx.load_model("tiny", device="cuda", download_root=save_dir)
			model = whisperx.load_model("large-v2", device="cuda", download_root=args['save_path'])
			result = model.transcribe(os.path.join(audio_path, audio_file))

			# Align words (optional, but useful for diarization)
			model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
			result_aligned = whisperx.align(result["segments"], model_a, metadata, os.path.join(audio_path, audio_file), "cuda")
			pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
			diarization = pipeline(os.path.join(audio_path, audio_file),min_speakers=args['min_speakers'], max_speakers=args['max_speakers'])

			# Add speaker labels to WhisperX segments
			result_segments = result_aligned["segments"]  # or result["segments"] if you skip alignment

			for segment in result_segments:
				segment_start = segment["start"]
				segment_end = segment["end"]
			
				 # Find the speaker active during this segment
				for turn, _, speaker in diarization.itertracks(yield_label=True):
					if turn.start <= segment_start <= turn.end or turn.start <= segment_end <= turn.end:
						segment["speaker"] = speaker
						segment["compound"] = scores["compound"]
						break
				else:
					segment["speaker"] = "unknown"
					segment["compound"] = scores["compound"]

			print(f"Saving {save_path}")
			np.save(save_path, result_segments, allow_pickle=True)
		except Exception as e:
			print(f"Error Processing {audio_file}")
			print(e)