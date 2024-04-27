import torch
import clip
import requests

from clipcap import ClipCaptionModel
from train import get_img_and_captions_paths
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

from PIL import Image

model_path = "clipcap-base-captioning-ft-hl-narratives/pytorch_model.pt" # change accordingly

# Load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
prefix_length = 10

# Load sample data
img_paths, captions = get_img_and_captions_paths("./dataset/samples.csv")

# Load ClipCap
model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
model.from_pretrained(model_path)
model = model.eval()
model = model.to(device)

# Load the image
img_url = 'https://datasets-server.huggingface.co/assets/michelecafagna26/hl-narratives/--/default/train/3/image/image.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# Extract the prefix
image = preprocess(raw_image).unsqueeze(0).to(device)
with torch.no_grad():
    prefix = clip_model.encode_image(image).to(
        device, dtype=torch.float32
    )
    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

# Generate the caption
model.generate_beam(embed=prefix_embed)[0]

# >> "He is riding a skateboard in a skate park, he wants to skate."
