import os

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

pretrained_model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

PROCESSED_TRANSCRIPTIONS_PATH = "data/transcriptions/processed/"

USER = "Steven Bartlett"

CONTEXT_LENGTH = 4096

MISTRAL_FORMAT_MESSAGE_ROW = lambda x: {"role": x["speaker"], "content": x["text"]}

MESSAGE_GROUPING_BUFFER = 175

EPISODES_TO_DROP = [
    "The Love Expert： Why Women Are Addicted To Toxic Men,＂Have A Boring Relationship Instead!＂ Logan Ury.csv",
    "The Coffee Expert： The Surprising Link Between Coffee & Your Mental Health! James Hoffmann.csv",
    "Derren Brown： UNLOCK The Secret Power Of Your Mind! ｜ E212.csv",
    "Sadhguru PREDICTION： Why We Are Now On ＂The Brink Of Extinction!＂.csv",
    "Psychology Expert： How Colours, Your First Name And Your Location Might Be Ruining Your Life!.csv",
]

EPISODES_TO_INVERT_SPEAKER = [
    "Chris Williamson： The Shocking New Research On Why Men And Women Are No Longer Compatible! ｜ E237.csv",
]


def invert_speaker(episode_transcript_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function inverts the speaker of each message in an episode transcript. The first row is dropped.

    Args:
        episode_transcript_df (DataFrame): A dataframe containing episode transcript data

    Returns:
        DataFrame: A dataframe containing episode transcript data with inverted speakers
    """
    speakers = episode_transcript_df["speaker"].unique().tolist()
    swap_speaker_map = dict(zip(speakers, speakers[::-1]))
    episode_transcript_df["speaker"] = episode_transcript_df["speaker"].map(
        swap_speaker_map
    )
    episode_transcript_df = episode_transcript_df.iloc[1:]

    return episode_transcript_df


def get_example_groups(episode_transcript_df, context_length, buffer=50):
    """
    This function groups messages in an episode transcript into an example until they exceed the context window of the model, and then starts a new example.

    The function requires the transcript dataframe `episode_transcript_df` which should have columns 'n_tokens' indicating the number of tokens in each message, and 'speaker' indicating the speaker of each message. The `context_length` parameter sets the maximum number of tokens in each group, and `buffer` allows for a buffer below this maximum.

    A buffer is included to account for any special tokens added by the tokenizer.

    The first speaker in the transcript must be 'user'.

    No single message can have more tokens than the threshold (context_length - buffer).

    Args:
        episode_transcript_df (DataFrame): A dataframe containing episode transcript data with columns 'n_tokens' and 'speaker'.
        context_length (int): The desired maximum length of a group in terms of tokens.
        buffer (int, optional): A buffer subtracted from context_length to define the actual threshold. Defaults to 10.

    Returns:
        list: A list of integers where each integer is the group identifier for the corresponding message in the episode_transcript_df.
    """

    n_tokens_list = episode_transcript_df["n_tokens"].to_list()
    speaker_list = episode_transcript_df["speaker"].to_list()

    cumsum = 0
    threshold = context_length - buffer
    group_id = 0
    groups = []

    assert speaker_list[0] == "user", "First speaker is not user"

    assert (
        max(n_tokens_list) < threshold
    ), "Max tokens in a message is greater than threshold"

    # group messages into groups of threshold tokens
    for i, (n_tokens, speaker) in enumerate(zip(n_tokens_list, speaker_list)):
        if cumsum + n_tokens <= threshold:
            cumsum += n_tokens
        else:
            if speaker == "user":
                group_id += 1
                cumsum = n_tokens
            else:
                group_id += 1
                groups[-1] = group_id
                cumsum = n_tokens + n_tokens_list[i - 1]

        groups.append(group_id)

    # check that max total tokens of any of the groups doesn't exceed threshold
    group_df = pd.DataFrame({"group": groups, "n_tokens": n_tokens_list})
    group_df = group_df.groupby("group").sum()
    assert (
        max(group_df["n_tokens"]) <= threshold
    ), "Max tokens in a group is greater than threshold"

    return groups


def create_examples_from_episode(
    episode_transcript, episode_file, tokenizer, format_message_row
):
    episode_transcript["text"] = episode_transcript["text"].str.strip()
    episode_transcript["text"] = episode_transcript["text"].str.replace("  ", " ")

    speaker_map = {
        n: "user" if n == USER else "assistant"
        for n in episode_transcript["speaker"].unique().tolist()
    }
    episode_transcript["speaker"] = episode_transcript["speaker"].map(speaker_map)

    episode_transcript["chat"] = episode_transcript.apply(format_message_row, axis=1)

    episode_transcript["n_tokens"] = episode_transcript["text"].apply(
        lambda text: len(tokenizer.encode(text))
    )

    episode_transcript["group_id"] = get_example_groups(
        episode_transcript, CONTEXT_LENGTH, MESSAGE_GROUPING_BUFFER
    )

    episode_examples = (
        episode_transcript.groupby("group_id")["chat"].apply(list).reset_index()
    )
    episode_examples.drop(columns=["group_id"], inplace=True)

    episode_examples_dataset = Dataset.from_pandas(episode_examples)
    episode_examples_dataset = episode_examples_dataset.map(
        lambda x: {
            "text": tokenizer.apply_chat_template(
                x["chat"], tokenize=False, add_generation_prompt=False
            )
        }
    )

    episode_examples = episode_examples_dataset.to_pandas()
    episode_examples.drop(columns=["chat"], inplace=True)
    episode_examples["episode"] = episode_file.split(".")[0]

    return episode_examples


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    all_examples = []

    for episode_file in os.listdir(PROCESSED_TRANSCRIPTIONS_PATH):
        if episode_file in EPISODES_TO_DROP:
            continue

        episode_transcript = pd.read_csv(PROCESSED_TRANSCRIPTIONS_PATH + episode_file)

        if episode_file in EPISODES_TO_INVERT_SPEAKER:
            episode_transcript = invert_speaker(episode_transcript)

        episode_examples = create_examples_from_episode(
            episode_transcript,
            episode_file,
            tokenizer=tokenizer,
            format_message_row=MISTRAL_FORMAT_MESSAGE_ROW,
        )

        all_examples.append(episode_examples)

    all_examples_df = pd.concat(all_examples, ignore_index=True)

    # find distinct episodes then randomly split into train, val. Create separate dataframes for each
    episodes = all_examples_df["episode"].unique().tolist()
    np.random.seed(42)
    train_episodes = np.random.choice(
        episodes, size=int(len(episodes) * 0.8), replace=False
    )
    val_episodes = [e for e in episodes if e not in train_episodes]
    train_examples_df = all_examples_df[all_examples_df["episode"].isin(train_episodes)]
    val_examples_df = all_examples_df[all_examples_df["episode"].isin(val_episodes)]

    os.makedirs("data/splits", exist_ok=True)
    train_examples_df.to_csv("data/splits/train.csv", index=False)
    val_examples_df.to_csv("data/splits/val.csv", index=False)


if __name__ == "__main__":
    main()
