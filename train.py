import torch
import torch.nn.functional as F
from dataset import Dataset
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
import clip

def custom_loss(similarity_score):
    # Define a margin threshold to penalize dissimilar captions
    margin = 0.1

    # Compute the loss based on the similarity score and the margin
    loss = F.relu(margin - similarity_score)

    return loss.mean()  # Return the mean of the losses across the batch

def train():
    # Instantiate Dataset
    dataset = Dataset()

    # Instantiate Model
    model = CLIPTextGenerator_multigpu()  # or CLIPTextGenerator() if not using multi-gpu

    # Define Optimizer
    optimizer = torch.optim.Adam(model.clip.parameters(), lr=0.001)

    num_epochs = 5
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.clip.train()
        total_loss = 0.0
        num_samples = 0
        for image_path, caption in dataset.data:
            optimizer.zero_grad()

            # Get Image Features
            image_features = model.get_img_feature([image_path], None)

            # Generate Captions
            output_captions = model.run(image_features, caption, beam_size=5)

            # Encode the target caption
            encoded_target_caption = clip.tokenize(caption).to(model.device)
            encoded_target_caption = model.clip.encode_text(encoded_target_caption)
            encoded_target_caption /= encoded_target_caption.norm(dim=-1, keepdim=True)

            for generated_caption in output_captions:
                # Encode the generated caption
                encoded_generated_caption = clip.tokenize(generated_caption).to(model.device)
                encoded_generated_caption = model.clip.encode_text(encoded_generated_caption)
                encoded_generated_caption /= encoded_generated_caption.norm(dim=-1, keepdim=True)

                # Compute similarity score
                similarity_score = (encoded_generated_caption @ encoded_target_caption.T).squeeze()

                # Calculate the loss
                loss = custom_loss(similarity_score)

                # Backpropagation
                loss.backward()

                # Update model parameters
                optimizer.step()

                total_loss += loss.item()
                num_samples += 1

        average_loss = total_loss / num_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}")

        # Save the best model based on validation loss
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.clip.state_dict(), "best_model.pth")

if __name__ == '__main__':
    train()
