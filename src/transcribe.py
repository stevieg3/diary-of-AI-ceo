import os
import logging

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import yt_dlp
import scrapetube
import whisperx
from dotenv import load_dotenv

load_dotenv('.env')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

YOUTUBE_VIDEO_BASE_URL = "https://www.youtube.com/watch?v="

DOAC_CHANNEL_URL = "https://www.youtube.com/@TheDiaryOfACEO"

AUDIO_OUTPUT_PATH = "data/audio"

AUDIO_EXT = 'mp3'

WHISPER_MODEL = "large-v2"

WHISPER_BATCH_SIZE = 16

WHISPER_COMPUTE_TYPE = 'float16'  # change to "int8" if low on GPU mem (may reduce accuracy)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LARGE_GAP_BETWEEN_DIALOGUE_SEC = 10


def download_audio_from_youtube(url: str, output_path: str, custom_filename:str=None) -> None:
    """
    Download audio from a YouTube video and save it to a specified path.
    :param url: YouTube video URL
    :param output_path: Path to save the audio file
    :param custom_filename: Custom filename for the audio file
    :return: None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define output format and path
    output_format = os.path.join(output_path, custom_filename + '.%(ext)s') if custom_filename else os.path.join(output_path, '%(title)s.%(ext)s')

    # Specify download options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_format,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': AUDIO_EXT,
            'preferredquality': '192',
        }],
    }

    # Download the audio file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_video_urls_from_channel(channel_url: str) -> list[str]:
    """
    Get all video URLs from a YouTube channel.
    :param channel_url: YouTube channel URL
    :return: List of video URLs
    """

    videos = scrapetube.get_channel(channel_url=channel_url)

    urls = []

    for video in videos:
        u = YOUTUBE_VIDEO_BASE_URL + video["videoId"]
        urls.append(u)

    logging.info(f"{len(urls)} video URLs in channel")

    return urls


def delete_files_in_directory(directory):
    """Delete all files in the specified directory.

    Args:
    directory (str): Path to the directory.

    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def main():
    urls = get_video_urls_from_channel(DOAC_CHANNEL_URL)

    # TODO: Drop sampling
    np.random.seed(42)
    urls = np.random.choice(urls, replace=False, size=5).tolist()

    for u in tqdm(urls):
        
        logging.info(f"Downloading audio from {u}")
        download_audio_from_youtube(url=u, output_path=AUDIO_OUTPUT_PATH)
        logging.info(f"Finished downloading audio from {u}")

        video_filename = os.listdir(AUDIO_OUTPUT_PATH)[0]
        video_title = video_filename.split('.' + AUDIO_EXT)[0]

        logging.info(f"Video title: {video_title}")

        audio_file = AUDIO_OUTPUT_PATH + '/' + video_filename

        # 1. Transcribe with original whisper (batched)
        logging.info("Transcribing audio with Whisper")
        logging.info(f"Device: {DEVICE}")
        model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=WHISPER_BATCH_SIZE)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

        # 3. Assign speaker labels
        logging.info("Adding speaker labels to transcription")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_ACCESS_TOKEN, device=DEVICE)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # 4. Put results into a DataFrame
        logging.info("Saving transcript as CSV")
        transcription_df = pd.DataFrame([{key: value for key, value in d.items() if key != 'words'} for d in result['segments']])
        
        # total_null_speaker_rows = transcription_df['speaker'].isnull().sum()
        # logging.info(f"Total null speaker rows: {total_null_speaker_rows}")
        # logging.info(transcription_df[transcription_df['speaker'].isnull()])
        # transcription_df.dropna(subset=['speaker'], inplace=True)
        # transcription_df.reset_index(drop=True, inplace=True)
        
        # # 5. Drop introduction by searching for first gap of more than X seconds between dialogue
        # transcription_df['time_to_next'] = transcription_df['start'].shift(-1) - transcription_df['end']
        # intro_end_idx = transcription_df[transcription_df['time_to_next'] > LARGE_GAP_BETWEEN_DIALOGUE_SEC].head(1).index.item()
        # transcription_df = transcription_df.iloc[intro_end_idx + 1:]

        # # 6. Group consecutive dialogue by same speaker into a single row
        # transcription_df['group'] = (transcription_df['speaker'] != transcription_df['speaker'].shift()).cumsum()
        # transcription_df['text'] = transcription_df['text'].apply(lambda x: x + ' ')

        # logging.info(f"transcription_df shape before groupby: {transcription_df.shape}")

        # transcription_df = transcription_df.groupby(['speaker', 'group']).agg(
        #     start=('start', 'first'),
        #     end=('end', 'last'),
        #     text=('text', 'sum')
        # ).reset_index()

        # transcription_df['duration'] = transcription_df['end'] - transcription_df['start']

        # transcription_df.sort_values('start', inplace=True)
        # transcription_df.drop(columns=['group'], inplace=True)
        # transcription_df.reset_index(drop=True, inplace=True)

        # logging.info(f"transcription_df shape after groupby: {transcription_df.shape}")

        # logging.info(transcription_df.head())

        # # 7. Check for overlapping timestamp intervals
        # intervals = pd.IntervalIndex.from_arrays(transcription_df['start'], transcription_df['end'], closed='both')

        # # Check for overlapping intervals excluding the interval itself
        # transcription_df['overlap'] = [any(interval != other_interval and interval.overlaps(other_interval) 
        #                     for other_interval in intervals) 
        #                 for interval in intervals]

        # # Filter rows that are part of an overlap
        # assert not any(transcription_df['overlap']), "Overlapping timestamp intervals found"

        # transcription_df.drop(columns=['overlap'], inplace=True)

        # # 8. Log first few dialogues
        # logging.info("Dialogue start...")

        # n = 10

        # for _, row in transcription_df.head(n).iterrows():
        #     words = row['text'].split()
        #     text = '\n'.join(' '.join(words[i:i+n]) for i in range(0, len(words), n))
        #     logging.info(f"{row['speaker']}:\n{text}\n")

        # 9. Save
        os.makedirs('data/transcriptions', exist_ok=True)
        transcription_df.to_csv('data/transcriptions' + '/' + video_title + '.csv', index=False)

        # 10. Delete audio
        logging.info("Deleting audio")
        delete_files_in_directory(AUDIO_OUTPUT_PATH)

        logging.info(f"Finished transcribing {video_title}")


if __name__ == "__main__":
    main()
