from training_ocr_qwen import train_dataset, data_collator, processor, lora_config
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model
import torch

batch=[train_dataset[i] for i in range(1)]
inputs = data_collator(batch)
print('Prepared inputs keys:', list(inputs.keys()))
print('Pixel values shape:', inputs.get('pixel_values').shape if 'pixel_values' in inputs else None)

print('Loading model (this may take a few minutes)')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    torch_dtype='auto',
    trust_remote_code=True,
    device_map={'': 'cuda:0'}
)
print('Resizing embeddings...')
model.resize_token_embeddings(len(processor.tokenizer))
print('Applying PEFT...')
model = get_peft_model(model, lora_config)
model.eval()

print('Running forward...')
with torch.no_grad():
    # Move tensor inputs to the same device as the model (device_map='auto' may place
    # the model on GPU/device shards). This prevents RuntimeError about tensors on CPU.
    model_device = next(model.parameters()).device
    print('Model primary device:', model_device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            print(f"Moving {k} device ->", v.device, "to", model_device)
            inputs[k] = v.to(model_device)
    out = model(**inputs)
print('Forward ok; output has logits:', hasattr(out, 'logits'))
