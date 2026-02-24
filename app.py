#!/usr/bin/env python3
"""
🚀 ODIA OCR SPACE - PRODUCTION VERSION
Robust Streamlit app for Odia text recognition using Qwen2.5-VL
"""

import streamlit as st
import os
import sys
import warnings
from contextlib import suppress

warnings.filterwarnings('ignore')

# ============ ENVIRONMENT SETUP ============
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Detect environment
IS_SPACES = os.environ.get('SPACES', False)
if IS_SPACES:
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Odia OCR - Qwen2.5-VL",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide streamlit UI elements in Spaces
if IS_SPACES:
    st.set_option('client.showErrorDetails', False)

# ============ TITLE & INFO ============
st.markdown("""
# 📖 OdiaLipi - Odia Text Recognition

**Advanced OCR with Qwen2.5-VL** 🚀

Fine-tuned on 145K+ Odia OCR samples
""")

st.info("""
✨ **Features:**
- 📊 **58% accuracy** on Odia OCR
- ⚡ **2.3 seconds** per image  
- 🎯 Qwen2.5-VL Vision-Language AI
- 🔒 Privacy-first processing
""")

st.divider()

# ============ MODEL LOADING WITH RETRY ============
@st.cache_resource(show_spinner=False)
def load_model():
    """Load Qwen OCR model with retry logic"""
    
    import torch
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        
        with st.spinner("⏳ Loading model (1-2 min on first run)..."):
            # Try loading our fine-tuned model
            try:
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    trust_remote_code=True,
                )
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "OdiaGenAIOCR/odia-ocr-qwen-finetuned",
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
                
                model.eval()
                return processor, model, device
                
            except Exception as e1:
                st.warning(f"⚠️ Could not load fine-tuned model: {str(e1)[:80]}")
                
                # Fallback: Load base model
                st.info("Loading base Qwen model instead...")
                processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    trust_remote_code=True,
                )
                
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                )
                
                model.eval()
                return processor, model, device
            
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)[:150]}")
        return None, None, None

# ============ TEXT EXTRACTION ============
def extract_text(image, processor, model, device):
    """Extract Odia text from image"""
    try:
        import torch
        from PIL import Image
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Prepare prompt
        prompt = "Read all text in this image. Return ONLY the text you can see."
        
        # Process
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )
        
        # Decode
        text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Post-process
        if "image" in text.lower():
            text = text.split("image")[-1].strip()
        
        return text.strip() if text.strip() else "⚠️ No text detected in image"
        
    except Exception as e:
        return f"❌ Error: {str(e)[:100]}"

# ============ MAIN UI ============
st.subheader("📸 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image with Odia or English text",
    type=["jpg", "jpeg", "png", "gif", "webp", "bmp"],
    help="Maximum recommended size: 10MB"
)

if uploaded_file:
    from PIL import Image
    import time
    
    st.divider()
    
    # Load image
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        image = None
    
    if image:
        # Display info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Image:**")
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("**Image Info:**")
            st.metric("Size", f"{image.width} × {image.height} px")
            st.metric("Format", image.format or "Unknown")
            st.metric("File Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        st.divider()
        
        # Extract button
        if st.button("🚀 Extract Text", use_container_width=True, type="primary"):
            
            # Load model
            processor, model, device = load_model()
            
            if processor and model and device:
                try:
                    start_time = time.time()
                    
                    with st.spinner("🔄 Extracting text..."):
                        text = extract_text(image, processor, model, device)
                    
                    elapsed = time.time() - start_time
                    
                    # Results
                    st.markdown("### ✅ Extraction Results")
                    
                    st.text_area(
                        "Extracted Text",
                        value=text,
                        height=150,
                        disabled=True,
                    )
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("⏱️ Processing Time", f"{elapsed:.2f}s")
                    col2.metric("📝 Characters", len(text))
                    col3.metric("📊 Words", len(text.split()))
                    
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)[:200]}")
            else:
                st.error("❌ Could not initialize model. Please refresh the page.")

else:
    st.markdown("""
    ### 📖 How to Use:
    1. **Upload** an image containing Odia or English text
    2. **Review** the preview
    3. **Click** "Extract Text"
    4. **Copy** your results
    
    ### ✨ Supported Formats:
    JPG, PNG, GIF, WebP, BMP (up to 200MB)
    
    ### 🎯 Model Information:
    - **Base Model:** Qwen/Qwen2.5-VL-3B-Instruct
    - **Fine-tuned on:** 145,000+ Odia OCR samples
    - **Training Method:** LoRA fine-tuning
    - **Benchmark Accuracy:** 58% on Odia text
    - **First Run:** ~2 minutes to download model
    
    ### ⚡ Performance:
    - **Speed:** 2-3 seconds per image
    - **GPU:** Automatic (CUDA/MPS/CPU)
    - **Memory:** ~8GB for model
    """)

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85em; margin-top: 40px;'>

**OdiaLipi** - Advanced Odia Text Recognition  
Powered by Qwen2.5-VL + LoRA Fine-tuning

[🔗 Model](https://huggingface.co/OdiaGenAIOCR/odia-ocr-qwen-finetuned) | 
[📚 Dataset](https://huggingface.co/datasets/shantipriya/odia-ocr-merged) |
[💝 Support](https://huggingface.co/OdiaGenAIOCR)

</div>
""", unsafe_allow_html=True)
