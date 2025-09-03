from fastapi import FastAPI, HTTPException
import os
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from mistralai.client import MistralClient
import io
import base64

# 1. Initialize models and clients once
app = FastAPI()

# Define a single function to hold your pipeline logic
def run_pipeline(prompt_text: str):
    # This API key is hardcoded directly in the source file.
    # This is not recommended for production applications.
    api_key = "j30zDyJtj4Lln6uXczoLP4kMeiniSdyv"
    
    # If using environment variables, this line would be used instead:
    # api_key = os.environ.get("MISTRAL_API_KEY") 
    
    if not api_key:
        raise HTTPException(status_code=500, detail="Mistral API key not configured.")

    model_name = "mistral-large-latest"
    client = MistralClient(api_key=api_key)

    # Story Generation
    messages = [
        {"role": "system", "content": "You are a master storyteller. Your only job is to provide a concise story based on the user's prompt. The story should be no more than 50 words. Do not add any conversational remarks or questions."},
        {"role": "user", "content": prompt_text}
    ]

    chat_response = client.chat(
        model=model_name,
        messages=messages,
        temperature=0.7 
    )
    story = chat_response.choices[0].message.content

    # Image Generation
    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    sd_pipe = sd_pipe.to("cuda")
    num_images = 3
    images = [sd_pipe(story).images[0] for _ in range(num_images)]

    # CLIP Scoring
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for i, img in enumerate(images):
        inputs = processor(text=story, images=img, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.item()
        scores.append((i, score))

    best_idx, best_score = max(scores, key=lambda x: x[1])
    best_image = images[best_idx]

    # Convert the best image to a base64 string for the web
    buffered = io.BytesIO()
    best_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image": img_str, "story": story}

@app.post("/generate_and_score")
def generate_and_score(data: dict):
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided.")

    try:
        result = run_pipeline(prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
