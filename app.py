#!/usr/bin/env python3
"""
🚀 ODIA OCR SPACE - GRADIO VERSION
Robust Gradio app for Odia text recognition using Qwen2.5-VL
Following Qaari-Urdu-OCR pattern with GPU optimization
"""

import gradio as gr
import time
import spaces
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import uuid
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ============ ENVIRONMENT SETUP ============
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Detect device for optimal dtype
def get_device_config():
    """Detect device and set appropriate dtype"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", torch.float32
    else:
        return "cpu", torch.float32

device, dtype = get_device_config()

# ============ MODEL LOADING ============
print("=" * 70)
print("🚀 ODIA OCR SPACE - LOADING MODEL")
print("=" * 70)

try:
    # Try loading fine-tuned model
    print("📥 Loading fine-tuned model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
        trust_remote_code=True
    )
    print("✅ Fine-tuned model loaded successfully")
    model_info = "Fine-tuned on 145K+ Odia OCR samples"
    
except Exception as e1:
    print(f"⚠️  Fine-tuned model loading failed: {str(e1)[:80]}")
    print("📥 Loading base model as fallback...")
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True
        )
        print("✅ Base model loaded as fallback")
        model_info = "Base Qwen2.5-VL model"
        
    except Exception as e2:
        print(f"❌ Failed to load any model: {e2}")
        raise RuntimeError("Could not load vision-language model")

model.eval()
max_tokens = 2000

print("=" * 70)
print(f"✅ Device: {device.upper()} | Dtype: {dtype}")
print(f"✅ Model: {model_info}")
print("=" * 70)

# ============ OCR PROCESSING ============
@spaces.GPU
def perform_ocr(image):
    """Process image and extract Odia text using OCR model"""
    
    try:
        # Validate input
        if image is None:
            return "❌ Error: No image provided"
        
        inputArray = np.any(image)
        if inputArray == False:
            return "❌ Error: Image is empty or invalid"
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image.astype('uint8')).convert('RGB')
        
        # Save with UUID for processing
        temp_file = str(uuid.uuid4()) + ".png"
        image_pil.save(temp_file)
        
        # OCR Prompt in Odia context
        prompt = """Below is an image of an Odia document or text. 
Please extract and return ONLY the plain text representation of the Odia content you see.
Do not add any explanations or hallucinations.
Return the text exactly as it appears in the document."""
        
        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{temp_file}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                use_cache=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Cleanup
        try:
            os.remove(temp_file)
        except:
            pass
        
        return output_text if output_text else "⚠️ No text extracted from image"
        
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"❌ OCR Error: {error_msg}")
        return f"❌ Error during processing: {error_msg}"

# ============ GRADIO INTERFACE ============
with gr.Blocks(title="OdiaLipi - Odia OCR with Qwen2.5-VL", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown(
        """
        # 📖 OdiaLipi - Odia Text Recognition
        **Advanced OCR with Qwen2.5-VL Vision-Language AI**
        
        Upload an image of Odia text to extract the content in real-time.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📸 Upload Image")
            image_input = gr.Image(
                type="numpy",
                label="Odia Document Image",
                sources=["upload", "webcam"]
            )
            
            # Models & formats info
            with gr.Accordion("📋 Supported Formats & Features", open=False):
                gr.Markdown("""
                **Image Formats:** JPG, PNG, GIF, WebP, BMP
                
                **Model:** Qwen2.5-VL-3B
                - 3 Billion parameters
                - Vision-Language capability
                - Context: up to 2000 output tokens
                - Device: GPU/MPS/CPU (auto-detected)
                
                **Fine-tuning:**
                - Dataset: 145K+ Odia OCR samples
                - Accuracy: ~58% on Odia text
                - Speed: ~2.3 seconds per image
                """)
            
            # Submit button
            submit_btn = gr.Button("🚀 Extract Text", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### 📝 Extracted Text")
            output = gr.Textbox(
                label="Odia Text",
                lines=20,
                show_copy_button=True,
                interactive=False
            )
            
            # Model information
            with gr.Accordion("ℹ️ Model Details", open=False):
                gr.Markdown(f"""
                **Model:** Qwen2.5-VL-3B
                **Status:** {model_info}
                **Device:** {device.upper()}
                **Precision:** {str(dtype).split('.')[-1]}
                
                **Capabilities:**
                - Extract Odia text from documents
                - Handwritten and printed text
                - Multi-line document processing
                - Real-time inference
                
                **Performance:**
                - First load: 2-3 minutes (model download)
                - Subsequent: <2 seconds (cached)
                """)
    
    # Establish processing flow
    submit_btn.click(
        fn=perform_ocr,
        inputs=image_input,
        outputs=output
    )
    image_input.change(
        fn=perform_ocr,
        inputs=image_input,
        outputs=output
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **OdiaLipi** • Powered by Qwen2.5-VL • 
        [Model Card](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned) • 
        [Dataset](https://huggingface.co/datasets/OdiaGenAIOCR/odia-ocr-merged)
        """
    )

if __name__ == "__main__":
    demo.launch()
