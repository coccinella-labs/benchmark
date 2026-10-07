import os
import time
from datetime import datetime

import torch
from huggingface_hub import HfApi
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from harpertoken.dataset import LiveSpeechDataset
from harpertoken.model import SpeechModel


def train_model(model_type="whisper", num_epochs=10, initial_lr=1e-4):
    # Initialize dataset
    dataset = LiveSpeechDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize model
    model = SpeechModel(model_type=model_type)
    optimizer = AdamW(model.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Training metrics
    training_log = {"loss": [], "learning_rate": [], "epoch_times": []}

    # Fine-tuning process
    model.train()
    print(f"Starting Speech Recognition AI Fine Tune with {model_type}...")
    print(f"Initial learning rate: {initial_lr}, Epochs: {num_epochs}")
    print("Press Ctrl+C to stop training")

    try:
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()
                inputs = batch["input_features"]

                # Create dummy labels for training
                labels = torch.zeros((1, 1), dtype=torch.long)

                # Forward pass
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()
                num_batches += 1

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

            # Update scheduler
            scheduler.step()

            # Calculate epoch metrics
            avg_loss = epoch_loss / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            training_log["loss"].append(avg_loss)
            training_log["learning_rate"].append(current_lr)
            training_log["epoch_times"].append(epoch_time)

            print(f"Epoch [{epoch + 1}/{num_epochs}] completed")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Epoch Time: {epoch_time:.2f} seconds\n")

    except KeyboardInterrupt:
        print("\nTraining stopped by user")

    # Save fine-tuned model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"models/speech_recognition_ai_fine_tune_{model_type}_{timestamp}"

    # Save model and training log
    model.model.save_pretrained(model_save_path)
    torch.save(training_log, os.path.join(model_save_path, "training_log.pt"))

    print(f"Fine-tuned model saved to {model_save_path}")
    print("Training metrics:")
    print(f"Final Loss: {training_log['loss'][-1]:.4f}")
    print(f"Final Learning Rate: {training_log['learning_rate'][-1]:.2e}")

    # Upload to Hugging Face Hub, opt-in via FT_UPLOAD.
    #
    # The target comes from FT_REPO_ID rather than being hardcoded. The previous
    # value, bniladridas/speech-recognition-ai-fine-tune-{model_type}, named a
    # repo that does not exist under any namespace, and the published models in
    # this org use short names, so a {model_type} template cannot map onto them.
    # There is no correct default to bake in, so a missing target is an error
    # rather than a guess.
    if os.getenv("FT_UPLOAD", "false").lower() != "true":
        print("Upload skipped. Set FT_UPLOAD=true with FT_REPO_ID to publish.")
    else:
        repo_id = os.getenv("FT_REPO_ID")
        if not repo_id:
            raise SystemExit(
                "FT_UPLOAD is set but FT_REPO_ID is empty. Set FT_REPO_ID to the "
                "target repository, for example harpertoken/talk."
            )
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise SystemExit("FT_UPLOAD is set but HF_TOKEN is empty.")

        api = HfApi(token=hf_token)
        print(f"Uploading model to Hugging Face Hub: {repo_id}")
        api.upload_folder(
            folder_path=model_save_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("Upload completed successfully!")
