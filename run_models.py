import time
from train import get_img_and_captions_paths
import clip
import torch
import matplotlib as plt
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from model.ZeroCLIP import CLIPTextGenerator
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from torchvision.transforms import functional as F

from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from clipcap import ClipCaptionModel
from train import get_img_and_captions_paths

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch
import clip
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

TEST_SAMPLES = "./test_images/test"

def run_clip_cap_model(use_train):
  # load clip
  device = "cuda" if torch.cuda.is_available() else "cpu"
  clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  prefix_length = 10

  if use_train:
    checkpoint = torch.load("clip_model_epoch_37.pt")
    clip_model.load_state_dict(checkpoint['model_state_dict'])

   # load ClipCap
  model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)
  list_image_path, list_caption = get_img_and_captions_paths(TEST_SAMPLES)


  model = model.eval()
  model = model.to(device)

  clip_cap_generated_captions = []
  for img_url in list_image_path:
    raw_image = Image.open(img_url).convert('RGB')

    # extract the prefix
    image = preprocess(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(
            device, dtype=torch.float32
        )
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

    # generate the caption
    print("Clip cap output" + model.generate_beam(embed=prefix_embed)[0])
    clip_cap_generated_captions.append(model.generate_beam(embed=prefix_embed)[0])

    return clip_cap_generated_captions, list_image_path, list_caption


def run_zero_clip_model(use_train):
  model = CLIPTextGenerator()
  list_image_path, list_caption = get_img_and_captions_paths(TEST_SAMPLES)
  zero_clip_generated_captions = []

  if use_train:
    checkpoint = torch.load("clip_model_epoch_37.pt")
    model.clip.load_state_dict(checkpoint['model_state_dict'])

  for image_path in list_image_path:
    image_features = model.get_img_feature([image_path], None)
    captions = model.run(image_features, "Image of", beam_size=5)

    encoded_captions = [model.clip.encode_text(clip.tokenize(c).to(model.device)) for c in captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

    print(captions)
    print('best clip:', "Image of" + captions[best_clip_idx])
    zero_clip_generated_captions.append(captions[best_clip_idx])

  return zero_clip_generated_captions



def calculate_scores(clip_cap_generated_captions, zero_clip_generated_captions, image_paths, ground_truth_captions):
  # BLEU score
  references = [caption.split(" ") for caption in ground_truth_captions]

  clip_cap_tokens = [caption.split(" ") for caption in clip_cap_generated_captions]
  zero_clip_tokens = [caption.split(" ") for caption in zero_clip_generated_captions]

  clip_cap_bleu_scores = []
  zero_clip_bleu_scores = []
  # clip cap scores
  for i, (reference, clip_cap_token) in enumerate(zip(references, clip_cap_tokens)):
    bleu_score = sentence_bleu([reference], clip_cap_token)
    print("Clip cap bleu score for pair", i + 1, ":", bleu_score)
    clip_cap_bleu_scores.append(bleu_score)

  # zero clip score
  for i, (reference, zero_clip_token) in enumerate(zip(references, zero_clip_tokens)):
    bleu_score = sentence_bleu([reference], zero_clip_token)
    print("Clip cap bleu score for pair", i + 1, ":", bleu_score)
    zero_clip_bleu_scores.append(bleu_score)


  # clipScore
  # Load and preprocess images
  images = []
  for path in image_paths:
      img = Image.open(path).convert("RGB")
      img = img.resize((224, 224))  # Resize image to expected size
      img_tensor = F.to_tensor(img)
      images.append(img_tensor)

  # Stack image tensors into a single tensor
  images_tensor = torch.stack(images)

  # Load CLIP model and processor
  clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

  clip_cap_clipScores = []
  zero_clip_clipScores = []
  # clip cap clipscore
  # Tokenize and encode captions and images
  inputs = clip_processor(text=clip_cap_generated_captions, images=images_tensor, return_tensors="pt", padding=True)
  with torch.no_grad():
      outputs = clip_model(**inputs)

  # Get embeddings
  caption_embeddings = outputs['text_features']
  image_embeddings = outputs['image_features']

  # Calculate cosine similarity between each caption and image pair
  for i, (caption, image) in enumerate(zip(caption_embeddings, image_embeddings)):
      similarity_score = cosine_similarity(caption.unsqueeze(0), image.unsqueeze(0)).item()
      print("Clip Cap CLIPScore for Pair", i+1, ":", similarity_score)
      clip_cap_clipScores.append(similarity_score)

  # zero clip clipscore
  # Tokenize and encode captions and images
  inputs = clip_processor(text=zero_clip_generated_captions, images=images_tensor, return_tensors="pt", padding=True)
  with torch.no_grad():
      outputs = clip_model(**inputs)

  # Get embeddings
  caption_embeddings = outputs['text_features']
  image_embeddings = outputs['image_features']

  # Calculate cosine similarity between each caption and image pair
  for i, (caption, image) in enumerate(zip(caption_embeddings, image_embeddings)):
      similarity_score = cosine_similarity(caption.unsqueeze(0), image.unsqueeze(0)).item()
      print("Clip Cap CLIPScore for Pair", i+1, ":", similarity_score)
      clip_cap_clipScores.append(similarity_score)

  return mean(clip_cap_bleu_scores), mean(clip_cap_clipScores), mean(zero_clip_bleu_scores), mean(zero_clip_clipScores)

def generate_graph(clip_cap_bleu_score, clip_cap_clipScore, zero_clip_bleu_score, zero_clip_clipScore):
   # Data
    categories = ['Clip Cap BLEU Score', 'Clip Cap ClipScore', 'Zero Clip BLEU Score', 'Zero Clip ClipScore']
    scores = [clip_cap_bleu_score, clip_cap_clipScore, zero_clip_bleu_score, zero_clip_clipScore]

    # Creating bar plot
    plt.bar(categories, scores, color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Scores')
    plt.ylabel('Values')
    plt.title('Comparison of Scores')

    # Saving as PDF
    plt.savefig('scores_comparison.pdf')

    # Displaying the plot
    plt.show()




if __name__ == '__main__':
  print("Running Clip Cap model", )
  clip_cap_generated_captions, list_image_path, list_caption = run_clip_cap_model(True)
  print("Running Zero Clip model")
  zero_clip_generated_captions = run_zero_clip_model(True)
  clip_cap_bleu_score, clip_cap_clipScore, zero_clip_bleu_score, zero_clip_clipScore = calculate_scores()
  generate_graph(clip_cap_bleu_score, clip_cap_clipScore, zero_clip_bleu_score, zero_clip_clipScore)
