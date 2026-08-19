"""
Experiment 11: AI-Based Content Generation System for Text, Image and Multimedia Applications
Course: CS4V48 - GenAI & LLM Laboratory
"""

import torch
from diffusers import StableDiffusionPipeline
from gtts import gTTS
from transformers import pipeline

def main():
    topic = "The benefits of renewable energy"
    print(f"Topic: '{topic}'\n")

    # 1. Text Generation
    print("--- Step 1: Text Generation (Flan-T5) ---")
    text_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    text_prompt = f"Write a short, engaging paragraph about: {topic}"
    generated_text = text_generator(text_prompt, max_length=80)[0]["generated_text"]
    print("Generated Text:\n", generated_text)
    print()

    # 2. Image Generation
    print("--- Step 2: Image Generation (Stable Diffusion) ---")
    image_prompt = f"An illustration representing {topic}, digital art"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch_dtype
    ).to(device)

    image = sd_pipe(image_prompt, num_inference_steps=25).images[0]
    image_path = "content_image.png"
    image.save(image_path)
    print(f"Image saved as '{image_path}'\n")

    # 3. Audio Generation (Text-to-Speech)
    print("--- Step 3: Audio Generation (gTTS) ---")
    tts = gTTS(text=generated_text, lang="en")
    audio_path = "content_audio.mp3"
    tts.save(audio_path)
    print(f"Audio saved as '{audio_path}'")

if __name__ == "__main__":
    main()
