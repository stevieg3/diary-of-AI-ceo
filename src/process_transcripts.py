import os
import logging
import re
import json

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


MINORITY_SPEAKER_PROPORTION_THRESHOLD = 0.01

HOST = "Steven Bartlett"


def clean_title(title: str) -> str:
    """
    Clean the title of the podcast episode to match the title of the transcription file.
    :param title: title of the podcast episode
    :return: cleaned title
    """
    title = re.sub(r'[^\w\s]', '', title)
    title = title.replace("  ", " ")
    title = title.replace(" ", "_")
    return title


def load_all_podcast_meta_data() -> dict:
    """
    Load the meta data for all podcast episodes.
    :return: dictionary of all podcast meta data
    """
    with open('data/podcast_meta_data.json') as f:
        podcast_meta_data = json.load(f)
        podcast_meta_data = {clean_title(k): v for k, v in podcast_meta_data.items()}
    return podcast_meta_data


def get_podcast_meta_data(file_name: str, all_podcast_meta_data: dict) -> tuple:
    """
    Get the meta data for the podcast episode.
    :param file_name: name of the transcription file
    :param all_podcast_meta_data: dictionary of all podcast meta data
    :return: tuple of guest name and intro end time
    """
    # remove mp3 extension from file name
    transcript_title = file_name.split(".")[0]
    transcript_title = clean_title(transcript_title)

    # get meta data for podcast episode
    meta_data = all_podcast_meta_data.get(transcript_title, None)

    if meta_data is None:
        return None, None
    
    guest_name = meta_data["guest"]
    intro_end_time = int(meta_data["intro_end_time"])

    logging.info(f"Podcast episode {transcript_title} found in podcast_meta_data.json. Guest name: {guest_name}. Intro end time: {intro_end_time}")

    return guest_name, intro_end_time


def strip_intro(transcript_df: pd.DataFrame, intro_end_time: int, large_gap_between_dialogue_sec=7) -> pd.DataFrame:
    """
    Strip the intro from the transcript.
    :param transcript_df: transcript dataframe
    :param intro_end_time: end time of the intro
    :return: transcript dataframe with intro stripped
    """
    logging.info(f"Stripping intro from transcript...")
    logging.info(f"Number of rows before removing intro: {len(transcript_df)}")

    transcript_df = transcript_df.copy()

    transcript_df['time_to_next'] = transcript_df['start'].shift(-1) - transcript_df['end']

    try:
        large_gap_idx = transcript_df[transcript_df['time_to_next'] > large_gap_between_dialogue_sec].head(1).index.item() + 1
    except:
        large_gap_idx = 999999
    
    chapter_end_idx = transcript_df[transcript_df["start"] > intro_end_time].head(1).index.item()

    if large_gap_idx < 5: # if large gap is in first 5 rows, use chapter end, since this is likely to be _during_ the intro
        cutoff_idx = chapter_end_idx
    else:
        cutoff_idx = min(large_gap_idx, chapter_end_idx)

    transcript_df = transcript_df.iloc[cutoff_idx:]

    logging.info(f"Number of rows after removing intro: {len(transcript_df)}")

    return transcript_df


def forward_fill_null_speakers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward fill null speaker entries in the DataFrame and log the number of null speaker entries.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.

    Returns:
    pd.DataFrame: DataFrame with null speaker entries forward filled.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_updated = df.copy()

    # Calculate and log the number of null speaker entries
    null_count = df_updated['speaker'].isna().sum()
    logging.info(f"Number of null speaker entries: {null_count}")

    # Forward fill null speaker entries
    df_updated['speaker'].ffill(inplace=True)

    return df_updated


def has_more_than_three_speakers(transcript_df: pd.DataFrame) -> bool:
    """
    Check if the DataFrame has more than three unique speakers.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.

    Returns:
    bool: True if more than three unique speakers are present, False otherwise.
    """
    unique_speakers = transcript_df['speaker'].nunique()
    return unique_speakers > 3


def minority_speaker_proportion(df: pd.DataFrame) -> tuple:
    """
    Calculate the proportion of rows spoken by the minority speaker and return the label of the minority speaker.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.

    Returns:
    tuple: Proportion of rows spoken by the minority speaker and the label of the minority speaker.
    """
    if df['speaker'].nunique() == 2:
        return 0., None

    speaker_proportions = df['speaker'].value_counts(normalize=True)
    minority_speaker = speaker_proportions.idxmin()
    minority_proportion = speaker_proportions.min()

    # Logging the proportions of rows by each speaker
    for speaker, proportion in speaker_proportions.items():
        logging.info(f"Speaker {speaker} proportion: {proportion:.2%}")

    return minority_proportion, minority_speaker


def is_first_speaker_minority(df: pd.DataFrame, minority_speaker: str) -> bool:
    """
    Check if the first speaker in the DataFrame is the minority speaker.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.
    minority_speaker (str): Label of the minority speaker.

    Returns:
    bool: True if the first speaker is the minority speaker, False otherwise.
    """
    first_speaker = df['speaker'].iloc[0]
    return first_speaker == minority_speaker


def forward_fill_minority_speaker_efficient(df: pd.DataFrame, minority_speaker: str) -> pd.DataFrame:
    """
    Forward fill the entries for the minority speaker with the label of the previous non-minority speaker.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.
    minority_speaker (str): Label of the minority speaker.

    Returns:
    pd.DataFrame: DataFrame with updated speaker labels.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_updated = df.copy()

    # Replace minority speaker labels with NaN
    df_updated.loc[df_updated['speaker'] == minority_speaker, 'speaker'] = None

    # Forward fill NaN values
    df_updated['speaker'].ffill(inplace=True)

    return df_updated


def assign_speakers(df: pd.DataFrame, host: str, guest: str) -> pd.DataFrame:
    """
    Assign the first speaker as the host and the second speaker as the guest.
    Assert that there are only two speakers in the DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing the episode data.
    host (str): Label to assign to the first speaker.
    guest (str): Label to assign to the second speaker.

    Returns:
    pd.DataFrame: DataFrame with updated speaker labels.
    """
    # Assert that there are only two unique speakers
    unique_speakers = df['speaker'].nunique()
    if unique_speakers != 2:
        raise ValueError("There should be only two speakers in the DataFrame.")

    # Identify the first and second speakers
    first_speaker = df['speaker'].iloc[0]
    second_speaker = df['speaker'].loc[df['speaker'] != first_speaker].iloc[0]

    # Replace speaker labels with host and guest
    df_updated = df.copy()
    df_updated['speaker'].replace({first_speaker: host, second_speaker: guest}, inplace=True)

    return df_updated


def aggregate_dialogue(transcript_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate dialogue by speaker
    
    Args:
    transcript_df (pd.DataFrame): DataFrame containing the episode data.
    
    Returns:
    pd.DataFrame: DataFrame with aggregated dialogue by speaker.
    """
    
    transcript_df['group'] = (transcript_df['speaker'] != transcript_df['speaker'].shift()).cumsum()
    transcript_df['text'] = transcript_df['text'].apply(lambda x: x + ' ')

    logging.info(f"transcript_df shape before groupby: {transcript_df.shape}")

    transcript_df = transcript_df.groupby(['speaker', 'group']).agg(
        start=('start', 'first'),
        end=('end', 'last'),
        text=('text', 'sum')
    ).reset_index()

    transcript_df['duration'] = transcript_df['end'] - transcript_df['start']

    transcript_df.sort_values('start', inplace=True)
    transcript_df.drop(columns=['group'], inplace=True)
    transcript_df.reset_index(drop=True, inplace=True)

    logging.info(f"transcript_df shape after groupby: {transcript_df.shape}")

    logging.info(transcript_df.head())

    # Check for overlapping timestamp intervals
    intervals = pd.IntervalIndex.from_arrays(transcript_df['start'], transcript_df['end'], closed='both')

    # Check for overlapping intervals excluding the interval itself
    transcript_df['overlap'] = [any(interval != other_interval and interval.overlaps(other_interval) 
                        for other_interval in intervals) 
                    for interval in intervals]

    # Filter rows that are part of an overlap
    assert not any(transcript_df['overlap']), "Overlapping timestamp intervals found"

    transcript_df.drop(columns=['overlap'], inplace=True)

    # Log first few dialogues
    logging.info("Dialogue start...")

    n = 20

    for _, row in transcript_df.head(5).iterrows():
        words = row['text'].split()
        text = '\n'.join(' '.join(words[i:i+n]) for i in range(0, len(words), n))
        logging.info(f"\n{row['speaker']}:\n{text}\n")

    return transcript_df


ALL_PODCAST_META_DATA = load_all_podcast_meta_data()


def main():
    # Get file names in data/transcriptions
    transcription_files = os.listdir("data/transcriptions")

    # Loop through files
    for file_name in transcription_files:

        logging.info(f"Processing file: {file_name}")
        
        guest_name, intro_end_time = get_podcast_meta_data(file_name, ALL_PODCAST_META_DATA)

        if intro_end_time is None:
            logging.info(f"Podcast episode {file_name.split('.')[0]} not found in podcast_meta_data.json. Skipping...")
            continue

        # Load transcript
        transcript_df = pd.read_csv(f"data/transcriptions/{file_name}")

        # Strip intro
        transcript_df = strip_intro(transcript_df, intro_end_time)

        # Forward fill null speakers (caused by numeric phrases)
        transcript_df = forward_fill_null_speakers(transcript_df)

        # Check if episode has more than 3 unique speakers
        if has_more_than_three_speakers(transcript_df):
            logging.info(f"Episode has more than 3 unique speakers. Skipping...")
            continue

        # Check if minority speaker proportion is above threshold
        minority_proportion, minority_speaker = minority_speaker_proportion(transcript_df)

        if minority_speaker:
            
            if minority_proportion > MINORITY_SPEAKER_PROPORTION_THRESHOLD:
                logging.info(f"Minority speaker proportion is above threshold. Skipping...")
                continue

            # Check if first speaker is minority speaker
            if is_first_speaker_minority(transcript_df, minority_speaker):
                logging.info(f"First speaker is minority speaker. Skipping...")
                continue

            # Forward fill minority speaker with previous speaker
            transcript_df = forward_fill_minority_speaker_efficient(transcript_df, minority_speaker)

        # Assign speakers
        transcript_df = assign_speakers(transcript_df, HOST, guest_name)

        # Aggregate dialogue
        transcript_df = aggregate_dialogue(transcript_df)

        transcript_df.to_csv(f"data/TEMP_{file_name.split('.')[0]}.csv", index=False)


if __name__ == "__main__":
    main()
