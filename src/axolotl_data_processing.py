import os
import json

import pandas as pd

train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')

combined_df = pd.concat([train_df, val_df], ignore_index=True)

# check for path data/axolotl, otherwise create it
os.makedirs('data/axolotl', exist_ok=True)

# Path for the output JSONL file
jsonl_file_path = 'data/axolotl/transcriptions.jsonl'

# Converting the CSV to JSONL format
with open(jsonl_file_path, 'w') as jsonl_file:
    for _, row in combined_df.iterrows():
        # Creating a JSON object with 'text' key
        json_object = {"text": row['text']}
        # Writing the JSON object as a line in the JSONL file
        jsonl_file.write(json.dumps(json_object) + '\n')
