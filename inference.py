#!/usr/bin/env python3
"""
Odia OCR Inference Script
==========================

Perform inference with the fine-tuned Qwen2.5-VL model.

Includes post-processing for clean Odia text extraction.
Supports single images, batch processing, and directory scanning.

Model: shantipriya/odia-ocr-qwen-finetuned
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import logging
from pathlib import Path
import argparse
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OdiaOCRInference:
    """Inference pipeline for Odia OCR."""

    def __init__(self, model_id="shantipriya/odia-ocr-qwen-finetuned"):
        """Initialize model and processor."""
        logger.info(f"Loading model: {model_id}")
        self.model_id = model_id

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @staticmethod
    def extract_odia_text(text):
        """Extract Odia Unicode characters (U+0B00-U+0B7F)."""
        odia_chars = [char for char in text if '\u0B00' <= char <= '\u0B7F']
        return ''.join(odia_chars)

    def transcribe(self, image_path):
        """Transcribe text from image (returns raw output with chat template)."""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path

            with torch.no_grad():
                inputs = self.processor(image, return_tensors="pt")
                output = self.model.generate(**inputs, max_new_tokens=256)
                raw_output = self.processor.decode(output[0], skip_special_tokens=True)

            return raw_output
        except Exception as e:
            logger.error(f"Error transcribing image: {e}")
            raise

    def process_image(self, image_path, return_raw=False):
        """Process image and return extracted Odia text."""
        try:
            raw_output = self.transcribe(image_path)
            odia_text = self.extract_odia_text(raw_output)

            if return_raw:
                return {"raw_output": raw_output, "odia_text": odia_text}
            return odia_text
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def batch_process(self, image_paths, return_raw=False):
        """Process multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, return_raw=return_raw)
                results.append({
                    "image": str(image_path),
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "image": str(image_path),
                    "error": str(e),
                    "success": False
                })
        return results

    def process_directory(self, directory, extensions=[".jpg", ".jpeg", ".png"], return_raw=False):
        """Process all images in a directory."""
        directory = Path(directory)
        image_files = []

        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images in {directory}")
        return self.batch_process(image_files, return_raw=return_raw)


def main():
    """CLI interface for inference."""
    parser = argparse.ArgumentParser(
        description="Odia OCR Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Single image
  python eval.py --image document.jpg

  # Directory of images
  python eval.py --directory ./images

  # Return raw output with chat template
  python eval.py --image document.jpg --raw

  # Save results to file
  python eval.py --directory ./images --output results.json
        """
    )

    parser.add_argument("--image", type=str, help="Path to single image", default=None)
    parser.add_argument("--directory", type=str, help="Path to image directory", default=None)
    parser.add_argument("--model", type=str, default="shantipriya/odia-ocr-qwen-finetuned",
                        help="Model ID")
    parser.add_argument("--raw", action="store_true", help="Return raw output with chat template")
    parser.add_argument("--output", type=str, help="Save results to JSON file", default=None)

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.directory:
        parser.print_help()
        return

    # Initialize
    ocr = OdiaOCRInference(args.model)

    # Process
    if args.image:
        logger.info(f"Processing image: {args.image}")
        result = ocr.process_image(args.image, return_raw=args.raw)
        if args.raw:
            logger.info(f"Raw Output:\n{result['raw_output']}\n")
            logger.info(f"Extracted Odia:\n{result['odia_text']}")
        else:
            logger.info(f"Extracted Odia:\n{result}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result if args.raw else {"odia_text": result}, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {args.output}")

    elif args.directory:
        logger.info(f"Processing directory: {args.directory}")
        results = ocr.process_directory(args.directory, return_raw=args.raw)

        for result in results:
            if result["success"]:
                logger.info(f"✅ {result['image']}")
                if args.raw:
                    logger.info(f"   Odia: {result['result']['odia_text'][:100]}...")
                else:
                    logger.info(f"   Text: {result['result'][:100]}...")
            else:
                logger.error(f"❌ {result['image']}: {result['error']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
