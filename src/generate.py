import os
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm

UNLIKELY_TOKEN = "ü"

TOKENIZER_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# GPT-4 generated prompts given examples of Steven Bartlett opening questions
PROMPTS = {
    "elon": "<s>[INST] Welcome, Elon. You're a figure often described as a visionary and a maverick, reshaping industries and even our approach to space exploration. But every story has a beginning. I'm curious about the earliest influences that shaped your perspective and drive. What were the pivotal moments or experiences in your early life that set you on this path of relentless innovation and ambition? [/INST]",
    "obama": "<s>[INST] Barack, as someone who has journeyed from the humble beginnings in Hawaii to the highest office in the land, your story is one of remarkable transformation and inspiration. I'm always fascinated by the early years that shape an individual. Could you take us back to those formative moments in your youth that set you on this incredible path, helping us understand the influences and experiences that molded you into the person you are today? [/INST]",
    "clarkson": "<s>[INST] Jeremy, welcome to the Diary of a CEO. You've become a household name through your work on 'Top Gear' and 'The Grand Tour,' captivating audiences with your unique take on the world of motoring. But I'm interested in the journey that led you here. Can you take me back to your earliest years, to the moments and influences that steered you towards this path of becoming one of the most recognized figures in the world of automotive journalism? [/INST]",
}


def parse_arguments():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Generate podcast transcripts.')

    # Add arguments
    parser.add_argument('--model_name_or_path', type=str, help='Model to load', required=True)
    parser.add_argument('--top_p', type=float, help='Top p', default=0.95)
    parser.add_argument('--top_k', type=int, help='Top k', default=0)
    parser.add_argument('--temperature', type=float, help='Temperature', default=0.7)
    parser.add_argument('--max_new_tokens', type=int, help='Max new tokens', default=4096)
    parser.add_argument('--seed', type=int, help='Seed', default=42)

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def load_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", load_in_4bit=True
    )
    return model


def main():
    
    args = parse_arguments()

    model_name_or_path = args.model_name_or_path
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    seed = args.seed

    set_seed(seed)

    model = load_model(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    unlikely_token_id = tokenizer.encode(UNLIKELY_TOKEN)[-1]

    for _, prompt in tqdm(PROMPTS.items()):

        model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens,
            eos_token_id=unlikely_token_id,  # to prevent early stopping at usual eos_token_id
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )

        transcript = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        model_name = model_name_or_path.replace("/", "-").replace("_", "-")

        filename = f"top_p={top_p}_top_k={top_k}_temperature={temperature}_max_new_tokens={max_new_tokens}_seed={seed}.txt"
        
        os.makedirs(f"generated_transcripts/{model_name}", exist_ok=True)
        
        with open(f"generated_transcripts/{model_name}/{filename}", "a") as f:
            f.write(transcript + '\n')


if __name__ == '__main__':
    main()
