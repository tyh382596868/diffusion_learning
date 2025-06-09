from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.nn import functional as F
from tqdm import tqdm


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
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    name = 'pokemon'
    image_size = 64
    dataset = get_dataset(name, image_size=image_size)

    if name in ['pokemon', 'smithsonian_butterflies_subset']:
        in_channels = 3
        out_channels = 3
    else:
        in_channels = 1
        out_channels = 1

    batch_size = 32*4
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet2DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        sample_size=image_size,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    num_epochs = 50
    losses = []

    for epoch in range(num_epochs):
        for batch in tqdm(train_dataloader):
            if name in ['pokemon', 'smithsonian_butterflies_subset']:
                clean_images = batch["images"].to(device)
            else:
                clean_images = batch[0].to(device)

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device).long()
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.autocast():
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if epoch % 5 == 0:
            model_for_pipeline = accelerator.unwrap_model(model)
            pipeline = DDPMPipeline(unet=model_for_pipeline, scheduler=scheduler)
            ims = pipeline(batch_size=4, output_type="pt").images

            rows, cols = 1, 4
            fig, axes = plt.subplots(rows, cols)
            axes = axes.flatten()
            for i, img in enumerate(ims):
                axes[i].imshow(img)
                axes[i].axis("off")

            plt.savefig(f'/data/tyh/trash/02_0_{name}_{image_size}_{epoch}_sampleViz.png')

    torch.save(accelerator.unwrap_model(model).state_dict(), f'/data/tyh/ws/diffusion_from_diffusers/weight/02_0_{name}_{image_size}_model_{epoch}.pth')

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

    model_for_pipeline = accelerator.unwrap_model(model)
    pipeline = DDPMPipeline(unet=model_for_pipeline, scheduler=scheduler)
    ims = pipeline(batch_size=4, output_type="pt").images

    rows, cols = 1, 4
    fig, axes = plt.subplots(rows, cols)
    axes = axes.flatten()
    for i, img in enumerate(ims):
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_sampleViz.png')
