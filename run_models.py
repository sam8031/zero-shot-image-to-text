import time
from train import image_caption_dataset, get_img_and_captions_paths
import clip
import torch

from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from clipcap import ClipCaptionModel
from train import get_img_and_captions_paths

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

import requests
from PIL import Image

model_path = "clipcap-base-captioning-ft-hl-narratives/pytorch_model.pt" # change accordingly

def run_clip_cap_model():
  # load clip
  device = "cuda" if torch.cuda.is_available() else "cpu"
  clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  prefix_length = 10

  # load sample data
  img_paths, captions = get_img_and_captions_paths("./dataset/samples.csv")

  # load ClipCap
  model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
  model.from_pretrained(model_path)
  model = model.eval()
  model = model.to(device)

  # load the image
  img_url = 'https://datasets-server.huggingface.co/assets/michelecafagna26/hl-narratives/--/default/train/3/image/image.jpg'
  raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

  # extract the prefix
  image = preprocess(raw_image).unsqueeze(0).to(device)
  with torch.no_grad():
      prefix = clip_model.encode_image(image).to(
          device, dtype=torch.float32
      )
      prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

  # generate the caption
  model.generate_beam(embed=prefix_embed)[0]

  # >> "He is riding a skateboard in a skate park, he wants to skate."

def run_zero_clip_model():
  image_dir = "dataset/images/"
  captions_file = "dataset/captions.txt"

  list_image_path, list_caption = get_img_and_captions_paths(captions_file, image_dir)
  model = CLIPTextGenerator_multigpu()

  checkpoint = torch.load("clip_model_epoch_1.pt")

  model.clip.load_state_dict(checkpoint['model_state_dict'])
  start_time = time.time()
  for image_path in list_image_path[:100]:
    image_features = model.get_img_feature([image_path], None)
    captions = model.run(image_features, "Image of", beam_size=5)

    encoded_captions = [model.clip.encode_text(clip.tokenize(c).to(model.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', "Image of" + captions[best_clip_idx])

  print('Time taken for epoch: {:.2f} seconds'.format(time.time() - start_time))

def compare_models():

if __name__ == '__main__':
  run_model()
