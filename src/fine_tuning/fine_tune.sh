python run_clm_no_trainer.py \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type linear \
    --num_warmup_steps 0 \
    --learning_rate 3e-4 \
    --lora_attention_dim 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_train_epochs 3 \
    --seed 42 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --output_dir /tmp/test-clm