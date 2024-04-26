import clip.model
import torch
import clip
import time
import torch.nn as nn
import csv

from torch import optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 216
EPOCH = 30
TRAIN_FILE = "./dataset/training.csv"
TEST_FILE = "./dataset/testing.csv"
IMAGE_DIR = "dataset/images/"

def get_img_and_captions_paths(file):
    # Extract image paths and captions from the given CSV file
    list_image_path = []
    list_captions = []
    with open(file, "r", newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter='|')
        next(reader)
        for row in reader:
            image_name, caption = row[0].strip(), row[2].strip()
            list_image_path.append(IMAGE_DIR + image_name)
            list_captions.append(caption)
    return list_image_path, list_captions

def validate(model, dataloader, loss_img, loss_caption, device):
    # Validate the model using the provided dataloader and calculate loss
    total_loss = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            images, captions = batch
            images = images.to(device)
            captions = captions.to(device)

            logits_per_image, logits_per_caption = model(images, captions)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            total_loss += (loss_img(logits_per_image, ground_truth) + loss_caption(logits_per_caption, ground_truth)) / 2

            progress_bar.set_postfix_str(f'Loss: {total_loss.item() / (batch_idx + 1):.5f}')

    average_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {average_loss:.5f}")

def train():
    # Load the CLIP model
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) # Must set jit=False for training

    class image_title_dataset(Dataset):
        def __init__(self, list_image_path,list_txt):

            self.image_path = list_image_path
            self.title  = clip.tokenize(list_txt, truncate=True) # You can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

        def __len__(self):
            return len(self.title)

        def __getitem__(self, idx):
            image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
            title = self.title[idx]
            return image,title

    # Get paths and captions for training and testing
    list_image_path, list_caption = get_img_and_captions_paths(TRAIN_FILE)

    # Load training data
    dataset = image_title_dataset(list_image_path, list_caption)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)

    # Load validation data
    val_image_paths, val_captions = get_img_and_captions_paths(TEST_FILE)
    val_dataset = image_title_dataset(val_image_paths, val_captions)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    if device == "cpu":
        model.float()

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-8,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # Training loop
    start_time = time.time()
    for epoch in range(EPOCH):
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{EPOCH}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            images,texts = batch

            images= images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else :
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            # Print statistics
            running_loss += total_loss.item()
            progress_bar.set_postfix_str(f'Epoch {epoch + 1}/{EPOCH}, Loss: {total_loss.item():.5f}, Percentage: {(batch_idx + 1) / len(train_dataloader) * 100:.2f}%')
            progress_bar.update()

        # Print average loss for the epoch
        print(f"Average Loss for Epoch {epoch + 1}: {running_loss / len(train_dataloader)}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"checkpoints/clip_model_epoch_{epoch + 1}.pt")

    print('Time taken for epoch: {:.2f} seconds'.format(time.time() - start_time))

    # Validation
    validate(model, val_dataloader, loss_img, loss_txt, device)

    progress_bar.clear()

if __name__ == "__main__":
    train()
