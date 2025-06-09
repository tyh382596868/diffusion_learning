from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
from torch.cuda.amp import autocast, GradScaler

def get_dataset(name, image_size=64):
    if name == 'smithsonian_butterflies_subset':
        dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        def transform(examples):
            examples = [preprocess(image) for image in examples["image"]]
            return {"images": examples}
        dataset.set_transform(transform)
        return dataset

    elif name == 'pokemon':
        dataset = load_dataset('huggan/pokemon', split='train')
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        def transform(examples):
            examples = [preprocess(image) for image in examples["image"]]
            return {"images": examples}
        dataset.set_transform(transform)
        return dataset

    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(
            root='/data/tyh/ws/diffusion_from_scrach/data',
            train=True,
            download=True,
            transform=transform
        )
        return dataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = 'pokemon'
    image_size = 512
    dataset = get_dataset(name, image_size=image_size)

    if name in ['pokemon', 'smithsonian_butterflies_subset']:
        in_channels = 3
        out_channels = 3
    elif name == 'mnist':
        in_channels = 1
        out_channels = 1

    batch_size = 8
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Visualize some data
    if name in ['pokemon', 'smithsonian_butterflies_subset']:
        batch = next(iter(train_dataloader))
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows, cols)
        axes = axes.flatten()
        for i, axis in enumerate(axes):
            if i < len(batch["images"]):
                img = batch["images"][i].permute(1, 2, 0) * 0.5 + 0.5
                img = torch.clamp(img, 0, 1).cpu().numpy()
                axis.imshow(img)
                axis.axis("off")
        os.makedirs('/data/tyh/ws/diffusion_from_diffusers/img', exist_ok=True)
        plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_datasetViz.png')

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)

    model = UNet2DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        sample_size=64,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    num_epochs = 50
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    scaler = GradScaler()
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_dataloader):
            clean_images = batch["images"].to(device) if name != 'mnist' else batch[0].to(device)
            noise = torch.randn_like(clean_images).to(device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            with autocast():
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  

            



        if epoch % 5 == 0:
            pipeline = DDPMPipeline(unet=model.module if isinstance(model, torch.nn.DataParallel) else model, scheduler=scheduler)
            ims = pipeline(batch_size=4, output_type="pt").images
            fig, axes = plt.subplots(1, 4)
            for i, img in enumerate(ims):
                axes[i].imshow(img)
                axes[i].axis('off')
            plt.savefig(f'/data/tyh/trash/02_0_{name}_{image_size}_{epoch}_sampleViz.png')

    # Save model and training results
    os.makedirs('/data/tyh/ws/diffusion_from_diffusers/weight', exist_ok=True)
    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), f'/data/tyh/ws/diffusion_from_diffusers/weight/02_0_{name}_{image_size}_model_{epoch}.pth')

    plt.subplots(1, 2, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Training step")
    plt.subplot(1, 2, 2)
    plt.plot(range(400, len(losses)), losses[400:])
    plt.title("Training loss from step 400")
    plt.xlabel("Training step")
    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_trainingLoss.png')

    pipeline = DDPMPipeline(unet=model.module if isinstance(model, torch.nn.DataParallel) else model, scheduler=scheduler)
    ims = pipeline(batch_size=4).images
    fig, axes = plt.subplots(1, 4)
    for i, img in enumerate(ims):
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_sampleViz.png')
