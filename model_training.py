import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Dataset class for POPE
# This benchmark tests whether vision-language models hallucinate objects in images
class POPEHallucinationDataset(Dataset):
    def __init__(self, annotation_file, image_folder, clip_processor):
        
        # Load all annotation data from JSON file
        with open(annotation_file, 'r') as file_handle:
            self.annotation_data = json.load(file_handle)
        
        self.image_folder = image_folder
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.annotation_data)

    def __getitem__(self, sample_idx):
        
        # Get the annotation entry for this sample
        annotation_entry = self.annotation_data[sample_idx]
        
        # Construct full path to the image file
        full_image_path = os.path.join(self.image_folder, annotation_entry['image_path'])
        
        # Process the image using CLIP's processor (resizes, normalizes, converts to tensor)
        processed_image = self.clip_processor(images=full_image_path, return_tensors="pt")['pixel_values'][0]
        
        # Get the question text (e.g., "Is there a dog in the image?")
        question_text = annotation_entry['question']
        
        # Convert label to float tensor for BCE loss (binary classification)
        ground_truth_label = torch.tensor(annotation_entry['label'], dtype=torch.float32)
        
        # Tokenize the question text into input IDs and attention masks
        text_inputs = self.clip_processor(text=question_text, return_tensors="pt", 
                                         padding=True, truncation=True)
        
        # Extract token IDs and attention mask (remove batch dimension with [0])
        text_token_ids = text_inputs['input_ids'][0]
        text_attention_mask = text_inputs['attention_mask'][0]
        
        return processed_image, text_token_ids, text_attention_mask, ground_truth_label


def collate_fn(batch_samples):
    
    # Unzip the batch into separate lists
    images, token_ids_list, attention_masks_list, labels = zip(*batch_samples)
    
    # Stack images (all same size after preprocessing)
    batched_images = torch.stack(images)
    
    # Pad token IDs to the longest sequence in the batch (pad with 0s)
    batched_token_ids = nn.utils.rnn.pad_sequence(token_ids_list, batch_first=True, padding_value=0)
    
    # Pad attention masks to match token IDs length (pad with 0s to ignore padded tokens)
    batched_attention_masks = nn.utils.rnn.pad_sequence(attention_masks_list, batch_first=True, padding_value=0)
    
    # Stack labels into a single tensor
    batched_labels = torch.stack(labels)
    
    return batched_images, batched_token_ids, batched_attention_masks, batched_labels


class POPEHallucinationDetector(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Load pretrained CLIP model (ViT-B/16 variant)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        # Get the dimensionality of CLIP's projection space (typically 512)
        clip_projection_dim = self.clip_model.config.projection_dim
        
        # Binary classifier: projects combined features to a single logit
        self.binary_classifier = nn.Linear(clip_projection_dim, 1)

    def forward(self, pixel_values, text_token_ids, text_attention_mask):
        
        # Encode image into CLIP's embedding space [batch_size, projection_dim]
        image_embeddings = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        # Encode text into CLIP's embedding space [batch_size, projection_dim]
        text_embeddings = self.clip_model.get_text_features(
            input_ids=text_token_ids, 
            attention_mask=text_attention_mask
        )
        
        # Combine image and text features via element-wise multiplication
        combined_features = image_embeddings * text_embeddings
        
        # Project to single logit for binary classification and remove last dimension
        classification_logits = self.binary_classifier(combined_features).squeeze(-1)
        
        return classification_logits


def train(model, train_dataloader, optimizer, loss_criterion, device):
    
    model.train()
    
    total_loss_sum = 0.0
    correct_predictions = 0
    total_samples = 0

    # Create progress bar for training batches
    progress_bar = tqdm(train_dataloader, desc="Training", ncols=80)
    
    for batch_images, batch_token_ids, batch_attention_masks, batch_labels in progress_bar:
        batch_images = batch_images.to(device)
        batch_token_ids = batch_token_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_labels = batch_labels.to(device)

        # Zero out gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        prediction_logits = model(batch_images, batch_token_ids, batch_attention_masks)
        
        # Calculate loss between predictions and ground truth
        batch_loss = loss_criterion(prediction_logits, batch_labels)
        
        # Backward pass: compute gradients
        batch_loss.backward()
        
        # Update model weights
        optimizer.step()

        # Track statistics for this batch
        current_batch_size = batch_images.size(0)
        total_loss_sum += batch_loss.item() * current_batch_size
        
        # Convert logits to binary predictions (threshold at 0.5 after sigmoid)
        binary_predictions = (torch.sigmoid(prediction_logits) > 0.5).float()
        correct_predictions += (binary_predictions == batch_labels).sum().item()
        total_samples += current_batch_size

        # Update progress bar with current metrics
        progress_bar.set_postfix({
            "loss": f"{batch_loss.item():.4f}",
            "acc": f"{(correct_predictions / total_samples):.4f}"
        })

    # Calculate average metrics over the entire epoch
    average_epoch_loss = total_loss_sum / total_samples
    average_epoch_accuracy = correct_predictions / total_samples
    
    return average_epoch_loss, average_epoch_accuracy


def evaluate(model, validation_dataloader, loss_criterion, device):

    model.eval()
    
    total_loss_sum = 0.0
    correct_predictions = 0
    total_samples = 0

    # Create progress bar for the validation batches
    progress_bar = tqdm(validation_dataloader, desc="Evaluating", ncols=80)
    
    # Disabling the gradient computation for efficiency during evaluation
    with torch.no_grad():
        for batch_images, batch_token_ids, batch_attention_masks, batch_labels in progress_bar:
            batch_images = batch_images.to(device)
            batch_token_ids = batch_token_ids.to(device)
            batch_attention_masks = batch_attention_masks.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass: get model predictions
            prediction_logits = model(batch_images, batch_token_ids, batch_attention_masks)
            
            # Calculating loss
            batch_loss = loss_criterion(prediction_logits, batch_labels)

            # Tracking statistics for this batch
            current_batch_size = batch_images.size(0)
            total_loss_sum += batch_loss.item() * current_batch_size
            
            # Convert logits to binary predictions
            binary_predictions = (torch.sigmoid(prediction_logits) > 0.5).float()
            correct_predictions += (binary_predictions == batch_labels).sum().item()
            total_samples += current_batch_size

            # Updating the progress bar with current metrics
            progress_bar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "acc": f"{(correct_predictions / total_samples):.4f}"
            })

    # Calculating the average metrics over the entire validation set
    average_epoch_loss = total_loss_sum / total_samples
    average_epoch_accuracy = correct_predictions / total_samples
    
    return average_epoch_loss, average_epoch_accuracy


def main():
    
    import argparse
    
    # Setting up the command-line argument parser
    argument_parser = argparse.ArgumentParser(
        description="Train a hallucination detector for vision-language models using POPE benchmark"
    )
    argument_parser.add_argument("--train_annotations", type=str, required=True, 
                                help="Path to training annotation JSON file")
    argument_parser.add_argument("--val_annotations", type=str, required=True, 
                                help="Path to validation annotation JSON file")
    argument_parser.add_argument("--image_folder", type=str, required=True, 
                                help="Directory containing image files")
    argument_parser.add_argument("--epochs", type=int, default=5, 
                                help="Number of training epochs (default: 5)")
    argument_parser.add_argument("--batch_size", type=int, default=32, 
                                help="Batch size for training and validation (default: 32)")
    argument_parser.add_argument("--lr", type=float, default=5e-5, 
                                help="Learning rate for optimizer (default: 5e-5)")
    argument_parser.add_argument("--device", type=str, 
                                default="cuda" if torch.cuda.is_available() else "cpu",
                                help="Device to use for training (cuda/cpu)")
    
    command_args = argument_parser.parse_args()

    # Initializing CLIP processor for image and text preprocessing 
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    # Creating the training and validation datasets
    training_dataset = POPEHallucinationDataset(
        command_args.train_annotations, 
        command_args.image_folder, 
        clip_processor
    )
    validation_dataset = POPEHallucinationDataset(
        command_args.val_annotations, 
        command_args.image_folder, 
        clip_processor
    )

    # Creating the dataloaders with custom collate function for padding
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=command_args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=command_args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initializing the model and moving to appropriate device
    hallucination_detector = POPEHallucinationDetector().to(command_args.device)
    
    # Initializing the AdamW optimizer
    model_optimizer = AdamW(hallucination_detector.parameters(), lr=command_args.lr)
    
    # Binary Cross Entropy loss with logits
    loss_function = nn.BCEWithLogitsLoss()

    # Tracking the best validation accuracy to save the best model
    best_validation_accuracy = 0.0
    
    # Main training loop over epochs
    for epoch_num in range(command_args.epochs):
        print("\n" + "=" * 50)
        print(f"EPOCH {epoch_num + 1}/{command_args.epochs}")
        print("=" * 50)

        # Train for one epoch
        train_loss, train_accuracy = train(
            hallucination_detector, 
            training_dataloader, 
            model_optimizer, 
            loss_function, 
            command_args.device
        )
        
        # Evaluate the validation set
        validation_loss, validation_accuracy = evaluate(
            hallucination_detector, 
            validation_dataloader, 
            loss_function, 
            command_args.device
        )

        # Print the epoch results
        print("\n📊 Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss:   {validation_loss:.4f}, Val Acc:   {validation_accuracy:.4f}")

        # Saving the model if it achieves best validation accuracy by far
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                hallucination_detector.state_dict(), 
                "best_pope_hallucination_detector.pth"
            )
            print(f"💾 Saved best model (Val Acc: {validation_accuracy:.4f})")


if __name__ == "__main__":
    main()