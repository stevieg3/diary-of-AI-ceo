import argparse
import logging

logging.basicConfig(level=logging.INFO)


def print_dialogue_turns(dialogue_list):
    """
    Prints each turn in the dialogue list and waits for the user to press Enter to continue.

    Parameters:
    dialogue_list (list): A list of dialogue strings.
    """
    input("Press Enter to start dialogue")

    for turn in dialogue_list:
        print(turn)
        input()  # Waits for Enter to be pressed without any prompt
    
    logging.info("End of dialogue")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_path", help="Path to the transcript file", required=True)
    args = parser.parse_args()

    transcript_path = args.transcript_path

    with open(transcript_path) as f:
        all_transcripts = f.readlines()

    logging.info(f"{len(all_transcripts)} transcripts found")

    for transcript in all_transcripts:
        transcript_lines = transcript.replace("[/INST]", "[/INST]\n").replace("</s>", "</s>\n").split("\n")
        print_dialogue_turns(transcript_lines)


if __name__ == "__main__":
    main()
