import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
train_file = "data/splits/train.csv"
validation_file = "data/splits/val.csv"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

data_files = {}
dataset_args = {}

data_files["train"] = train_file

data_files["validation"] = validation_file

extension = train_file.split(".")[-1]

if extension == "txt":
    extension = "text"
    dataset_args["keep_linebreaks"] = True

data = load_dataset(extension, data_files=data_files, **dataset_args)

tokenizer = AutoTokenizer.from_pretrained(model_id) # , use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

column_names = data["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

data = data.map(
    tokenize_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

# data = load_dataset("Abirate/english_quotes")
# data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()



