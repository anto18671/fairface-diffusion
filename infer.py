# ============================================
#  FairFace Conditioned Diffusion Inference
#  Load epochs 10-90, 4 samples each
#  Saves image grid with row per epoch
# ============================================

# ---------------------------
#  Import libraries
# ---------------------------
import os
import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch import nn
from PIL import ImageDraw, ImageFont

# ---------------------------
#  Load UNet model
# ---------------------------
def load_model(checkpoint_path, image_size, device):
    # Create model
    model = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=512
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    return model

# ---------------------------
#  Generate samples for epoch
# ---------------------------
def generate_samples(unet, cond_emb, device, size, num_samples=4):
    # Create DDPM scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Container for images
    samples = []

    # Generate samples
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for i in range(num_samples):
            cond = torch.tensor([i], device=device)
            emb = cond_emb(cond).unsqueeze(1)

            x = torch.randn(1, 3, size, size).to(device)

            for t in reversed(range(1000)):
                t_tensor = torch.tensor([t], device=device)
                pred = unet(x, t_tensor, encoder_hidden_states=emb).sample
                x = scheduler.step(pred, t, x).prev_sample

            img = x.clamp(-1, 1).add(1).div(2).squeeze().cpu()
            samples.append(img)

    return torch.stack(samples)

# ---------------------------
#  Create image grid
# ---------------------------
def create_grid(image_rows, image_size, output_path):
    # Calculate grid dimensions
    num_rows = len(image_rows)
    num_cols = len(image_rows[0])

    # Create image grid
    grid = torchvision.utils.make_grid(
        torch.cat(image_rows, dim=0),
        nrow=num_cols,
        padding=2
    )

    # Convert to PIL image
    pil_img = transforms.ToPILImage()(grid)

    # Add epoch labels on the left
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for i, epoch in enumerate(range(10, 100, 10)):
        y = i * (image_size + 2)
        draw.text((2, y + 2), f"Epoch {epoch}", fill="white", font=font)

    # Save image
    pil_img.save(output_path)

# ---------------------------
#  Inference main logic
# ---------------------------
def inference():
    # Set config
    image_size = 128
    output_dir = "./assets"
    checkpoints_dir = "models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup conditional embedding
    num_conditions = 9 * 2 * 7
    cond_emb = nn.Embedding(num_conditions, 512).to(device)
    cond_emb.eval()

    # Store all rows of generated samples
    all_samples = []

    # Iterate over epochs 10 to 90
    for epoch in range(10, 100, 10):
        # Load model
        ckpt_path = os.path.join(checkpoints_dir, f"unet_epoch_{epoch}.pt")
        unet = load_model(ckpt_path, image_size, device)

        # Generate 4 samples for current epoch
        samples = generate_samples(unet, cond_emb, device, image_size, num_samples=4)
        all_samples.append(samples)

    # Create and save final grid
    create_grid(all_samples, image_size, os.path.join(output_dir, "grid_epochs_10_90.png"))

# ---------------------------
#  Run inference
# ---------------------------
if __name__ == "__main__":
    inference()
