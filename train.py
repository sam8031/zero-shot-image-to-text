import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from model.ZeroCLIP import CLIPTextGenerator
from torch import nn
import clip

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.clip.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        image_paths = dataloader.dataset.data[0]
        captions = dataloader.dataset.data[1]
        optimizer.zero_grad()
        generated_captions = []

        for image_path in image_paths:
            # Forward pass
            image_features = model.get_img_feature([image_path], None)
            condition_text = captions[image_paths.index(image_path)].split(" ")[0]
            print(f"Catption: {captions[image_paths.index(image_path)]} Condition text: {condition_text}")
            generated_caption = model.run(image_features,condition_text, beam_size=1)
            generated_captions.append(generated_caption)

        # Assuming 'generated_captions' and 'captions' are lists of tokenized captions (strings)
        # Convert tokenized captions into numerical representations (indices)
        generated_captions_tensor = [model.lm_tokenizer.convert_tokens_to_ids(clip.tokenize(c).to(model.device)) for c in generated_captions]
        captions_tensor = [model.lm_tokenizer.convert_tokens_to_ids(clip.tokenize(c).to(model.device)) for c in captions]

        # Compute loss
        loss = criterion.forward(torch.tensor(generated_captions_tensor), torch.tensor(captions_tensor))

        # Backpropagation
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        epoch_loss = total_loss / len(image_paths)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    dataset = Dataset()

    # Instantiate Model
    model = CLIPTextGenerator_multigpu()

    criterion = CaptionLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.clip.parameters(), lr=0.001)

    # Train the model
    train(model, dataset, optimizer, criterion, num_epochs=1)
