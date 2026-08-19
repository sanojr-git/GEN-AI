"""
Experiment 08: Image Generation Application Using Diffusion Models
Course: CS4V48 - GenAI & LLM Laboratory
"""

import torch
from diffusers import StableDiffusionPipeline

def main():
    print("Loading Stable Diffusion v1-5 pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(device)

    prompt = "A futuristic city skyline at sunset, digital art, highly detailed"
    print(f"Generating image for prompt: '{prompt}'")

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    output_path = "generated_city.png"
    image.save(output_path)
    print(f"Image generated and saved as '{output_path}'")

if __name__ == "__main__":
    main()
