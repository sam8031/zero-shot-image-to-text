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
EPOCH = 50
TRAIN_FILE = "./dataset/training.csv"
IMAGE_DIR = "dataset/images/"

def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

class CaptionCLIP(nn.Module):
    def __init__(self, model) :
        super(CaptionCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

class image_caption_dataset(Dataset):
    def __init__(self, list_image_path, list_captions, preprocess):

        self.image_path = list_image_path
        self.captions  = clip.tokenize(list_captions)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        caption = self.captions[idx]
        return image,caption

def get_img_and_captions_paths():
    list_image_path = []
    list_captions = []
    with open(TRAIN_FILE, "r", newline='') as csvFile:
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

def train():
    # Define the root directory and captions file


    # Load the CLIP model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    model_caption = CaptionCLIP(model)
    model_image = ImageCLIP(model)

    model_caption = nn.DataParallel(model_caption)
    model_image = nn.DataParallel(model_image)

    list_image_path, list_caption = get_img_and_captions_paths()


    dataset = image_caption_dataset(list_image_path, list_caption, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    loss_img = nn.CrossEntropyLoss()
    loss_caption = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-6, betas=(0.9,0.98), eps=1e-6, weight_decay=0.05)

    # Training loop
    model.train()
    start_time = time.time()
    for epoch in range(EPOCH):
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{EPOCH}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            images, captions = batch

            image_embedding = model_image(images)
            caption_embedding = model_caption(captions)

            logit_scale = model.logit_scale.exp()
            logits_per_image, logits_per_caption = create_logits(image_embedding,caption_embedding,logit_scale)
            ground_truth = ground_truth = torch.arange(BATCH_SIZE).to(device)

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

    # Save the model
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss,
    },f"clip_model_epoch_{epoch + 1}.pt")

    print('Time taken for epoch: {:.2f} seconds'.format(time.time() - start_time))
    progress_bar.clear()

if __name__ == "__main__":
    train()