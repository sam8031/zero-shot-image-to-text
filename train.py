import clip.model
import torch
import clip
from torch.optim import Adam
import time
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import csv

BATCH_SIZE = 32
EPOCH = 32
TRAIN_FILE = "./dataset/training.csv"
TEST_FILE = "./dataset/testing.csv"
IMAGE_DIR = "dataset/images/"

class image_caption_dataset(Dataset):
    def __init__(self, list_image_path, list_caption, preprocess):

        self.image_path = list_image_path
        self.captions  = clip.tokenize(list_caption, truncate=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        caption = self.captions[idx]
        return image,caption

def get_img_and_captions_paths(file):
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

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def validate(model, dataloader, loss_img, loss_caption, device):
    model.eval()  # Set the model to evaluation mode
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    list_image_path, list_caption = get_img_and_captions_paths(TRAIN_FILE)

    # load training data
    dataset = image_caption_dataset(list_image_path, list_caption, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # load validation data
    val_image_paths, val_captions = get_img_and_captions_paths(TEST_FILE)
    val_dataset = image_caption_dataset(val_image_paths, val_captions, preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_caption = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # Training loop
    start_time = time.time()
    model.train()
    for epoch in range(EPOCH):
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{EPOCH}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            images, captions = batch

            images= images.to(device)
            captions = captions.to(device)

            logits_per_image, logits_per_caption = model(images, captions)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_caption(logits_per_caption, ground_truth)) / 2
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
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
    validate(model, val_dataloader, loss_img, loss_caption, device)

    progress_bar.clear()


if __name__ == "__main__":
    train()