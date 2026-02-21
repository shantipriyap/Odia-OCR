from training_ocr_qwen import train_dataset, eval_dataset, data_collator, processor, lora_config
from transformers import Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from peft import get_peft_model
import torch

print('Loading model (smoke-run) on cuda:0')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    torch_dtype='auto',
    trust_remote_code=True,
    device_map={'': 'cuda:0'},
)
print('Resizing embeddings...')
model.resize_token_embeddings(len(processor.tokenizer))
print('Applying PEFT...')
model = get_peft_model(model, lora_config)

smoke_args = TrainingArguments(
    output_dir='./qwen_ocr_smoke',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    max_steps=2,
    logging_steps=1,
    save_strategy='no',
    fp16=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=smoke_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print('Starting smoke training...')
trainer.train()
print('Smoke training finished')
