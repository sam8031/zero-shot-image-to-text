import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from model.ZeroCLIP import CLIPTextGenerator
from torch import nn
import clip
import time
import argparse

class CaptionLoss(nn.Module):
    def __init__(self):
        super(CaptionLoss, self).__init__()

    def forward(self, generated_captions, ground_truth_captions):
        # Compute the loss between the generated captions and ground truth captions
        # For example, you could use cosine similarity or any other suitable metric
        # Here, let's use negative cosine similarity as a loss
        generated_captions = generated_captions.float()
        ground_truth_captions = ground_truth_captions.float()
        cosine_similarity = torch.nn.functional.cosine_similarity(generated_captions, ground_truth_captions)
        loss = -torch.mean(cosine_similarity)
        return loss

def train(model, dataset, optimizer, criterion: CaptionLoss, num_epochs=5, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.clip.train()

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0.0
        image_paths = dataloader.dataset.data[0]
        captions = dataloader.dataset.data[1]
        for image_path in image_paths:
            # Forward pass
            image_features = model.get_img_feature([image_path], None)
            curr_captions =" ".join(captions[:5])
            print(f"Target Captions: {curr_captions} Cond Test: {captions[0][0].lower()}")
            generated_captions = model.run(image_features, captions[0][0].lower() , beam_size=5)
            target_captions = captions[:5]
            captions = captions[5:]

            optimizer.zero_grad()
            # Assuming 'generated_captions' and 'captions' are lists of tokenized captions (strings)
            # Convert tokenized captions into numerical representations (indices)
            generated_captions_tensor = [model.lm_tokenizer.convert_tokens_to_ids(clip.tokenize(c).to(model.device)) for c in generated_captions]
            captions_tensor = [model.lm_tokenizer.convert_tokens_to_ids(clip.tokenize(c).to(model.device)) for c in target_captions]

            # Compute loss
            loss = criterion.forward(torch.tensor(generated_captions_tensor), torch.tensor(captions_tensor))

            # Backpropagation
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(image_paths)
        end_time = time.time()  # End timer for epoch
        epoch_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')

def run(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    image_paths = dataloader.dataset.data[0]
    for image_path in image_paths:
        image_features = model.get_img_feature([image_path], None)
        captions = model.run(image_features, "Image of", beam_size=5)

        encoded_captions = [model.clip.encode_text(clip.tokenize(c).to(model.device)) for c in captions]
        encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
        best_clip_idx = (torch.cat(encoded_captions) @ image_features.t()).squeeze().argmax().item()

        end_time = time.time()
        total_time = end_time - start_time
        print(captions)
        print('best clip:', "Image of" + captions[best_clip_idx] )
        print(f"Time: {total_time:.2f} seconds")



def main():
    parser = argparse.ArgumentParser(description='Train or run CLIP model.')
    parser.add_argument('mode', choices=['train', 'run', 'multi-train', 'multi-run'], help='Mode for training or running the model')
    args = parser.parse_args()

    if args.mode in ['train', 'multi-train']:
        dataset = Dataset(80, "dataset/training_captions.txt")
        model = CLIPTextGenerator_multigpu() if args.mode == 'multi-train' else CLIPTextGenerator()
        criterion = CaptionLoss()
        optimizer = torch.optim.Adam(model.clip.parameters(), lr=0.3)
        train(model, dataset, optimizer, criterion, num_epochs=1)
        # Save the trained model
        torch.save(model.clip.state_dict(), 'trained_model.pth')
    elif args.mode in ['run', 'multi-run']:
        # Load the trained model
        dataset = Dataset(20, "dataset/testing_caption.txt")
        model = CLIPTextGenerator_multigpu() if args.mode == 'multi-run' else CLIPTextGenerator()
        # model.clip.load_state_dict(torch.load('trained_model.pth'))
        run(model, dataset)
    else:
        print('Invalid mode')

if __name__ == '__main__':
    main()
