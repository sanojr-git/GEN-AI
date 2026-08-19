"""
Experiment 09: Multimodal AI Application Integrating Text and Image Inputs
Course: CS4V48 - GenAI & LLM Laboratory
"""

import requests
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
)

def main():
    image_url = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
    print(f"Fetching sample image from: {image_url}")
    raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

    # ---------- 1. Image Captioning ----------
    print("\n--- Image Captioning (BLIP) ---")
    cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = cap_processor(raw_image, return_tensors="pt")
    caption_ids = cap_model.generate(**inputs, max_new_tokens=30)
    caption = cap_processor.decode(caption_ids[0], skip_special_tokens=True)
    print("Generated Caption:", caption)

    # ---------- 2. Visual Question Answering ----------
    print("\n--- Visual Question Answering (BLIP VQA) ---")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    question = "What animal is in the picture?"
    vqa_inputs = vqa_processor(raw_image, question, return_tensors="pt")
    answer_ids = vqa_model.generate(**vqa_inputs)
    answer = vqa_processor.decode(answer_ids[0], skip_special_tokens=True)

    print("Question:", question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
