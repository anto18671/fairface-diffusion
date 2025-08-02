# ============================================
#  FairFace Conditioned Diffusion Trainer
#  Load from HuggingFace Datasets (0.25 + 1.25)
#  Merged train+val, Discrete Conditioning
#  Save & sample every epoch
# ============================================

# ---------------------------
#  Import libraries
# ---------------------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from datasets import load_dataset

# ---------------------------
#  Count model parameters
# ---------------------------
def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}, Trainable: {trainable:,}")

# ---------------------------
#  Custom FairFace Dataset (streamed)
# ---------------------------
class FairFaceConditionedDataset(Dataset):
    # Initialize dataset
    def __init__(self, hf_dataset, image_size):
        self.dataset = hf_dataset
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Extract discrete label maps once
        all_ages = sorted(set(self.dataset.unique("age")))
        all_genders = sorted(set(self.dataset.unique("gender")))
        all_races = sorted(set(self.dataset.unique("race")))

        self.age2idx = {v: i for i, v in enumerate(all_ages)}
        self.gender2idx = {v: i for i, v in enumerate(all_genders)}
        self.race2idx = {v: i for i, v in enumerate(all_races)}

    # Get sample
    def __getitem__(self, idx):
        row = self.dataset[idx]

        if row["image"] is None:
            return self.__getitem__((idx + 1) % len(self))

        image = row["image"].convert("RGB")
        tensor = self.transform(image)

        age_id = self.age2idx[row["age"]]
        gender_id = self.gender2idx[row["gender"]]
        race_id = self.race2idx[row["race"]]
        cond_id = (age_id * 14) + (gender_id * 7) + race_id

        return tensor, cond_id

    # Get dataset length
    def __len__(self):
        return len(self.dataset)

# ---------------------------
#  Save generated samples
# ---------------------------
def save_samples(unet, cond_emb, device, epoch, out_dir, size):
    os.makedirs(out_dir, exist_ok=True)
    unet.eval()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(3):
            cond = torch.tensor([i], device=device)
            emb = cond_emb(cond).unsqueeze(1)

            x = torch.randn(1, 3, size, size).to(device)
            scheduler = DDPMScheduler(num_train_timesteps=1000)

            for t in reversed(range(1000)):
                t_tensor = torch.tensor([t], device=device)
                pred = unet(x, t_tensor, encoder_hidden_states=emb).sample
                x = scheduler.step(pred, t, x).prev_sample

            img = x.clamp(-1, 1).add(1).div(2).squeeze().cpu()
            transforms.ToPILImage()(img).save(f"{out_dir}/sample_epoch_{epoch}_{i}.png")

# ---------------------------
#  Main training loop
# ---------------------------
def train():
    # Set config
    image_size = 128
    batch_size = 24
    lr = 1e-4
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)

    # Load and merge dataset splits (no in-memory conversion)
    print("Loading FairFace splits (0.25 + 1.25 train+val)...")

    dataset_025_train = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train")
    dataset_025_val = load_dataset("HuggingFaceM4/FairFace", "0.25", split="validation")
    dataset_125_train = load_dataset("HuggingFaceM4/FairFace", "1.25", split="train")
    dataset_125_val = load_dataset("HuggingFaceM4/FairFace", "1.25", split="validation")

    # Merge splits using HuggingFace Dataset's built-in tools
    from datasets import concatenate_datasets
    merged_dataset = concatenate_datasets([
        dataset_025_train,
        dataset_025_val,
        dataset_125_train,
        dataset_125_val
    ])

    # Prepare dataloader
    dataset = FairFaceConditionedDataset(merged_dataset, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Setup model
    num_conditions = 9 * 2 * 7
    cond_emb = nn.Embedding(num_conditions, 512).to(device)
    unet = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=512
    ).to(device)

    count_model_parameters(unet)

    # Setup training
    optimizer = torch.optim.AdamW(list(unet.parameters()) + list(cond_emb.parameters()), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Starting epoch {epoch}/{num_epochs} ---")
        pbar = tqdm(dataloader, dynamic_ncols=True)

        for images, cond_ids in pbar:
            images = images.to(device)
            cond_ids = cond_ids.to(device)

            with torch.amp.autocast("cuda"):
                emb = cond_emb(cond_ids).unsqueeze(1)
                t = torch.randint(0, 1000, (images.size(0),), device=device).long()
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(images, noise, t)
                pred = unet(noisy, t, encoder_hidden_states=emb).sample
                loss = F.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Save checkpoint and samples after epoch
        torch.save(unet.state_dict(), f"{output_dir}/unet_epoch_{epoch}.pt")
        save_samples(unet, cond_emb, device, epoch, output_dir, image_size)

# ---------------------------
#  Run training
# ---------------------------
if __name__ == "__main__":
    train()
