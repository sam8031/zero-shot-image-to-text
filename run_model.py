import time
import clip
import torch

from train import image_caption_dataset, get_img_and_captions_paths
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu

def run_model():
    # Run the CLIP text generation model on a subset of images and print generated captions
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

if __name__ == '__main__':
  run_model()
