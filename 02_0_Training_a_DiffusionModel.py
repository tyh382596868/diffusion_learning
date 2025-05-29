from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline
from tqdm import tqdm
if __name__=="__main__":
    device = 'cuda'
    dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    print(dataset)
    from torchvision import transforms
    image_size = 64
    # Define transformations
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize
            transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
            transforms.ToTensor(),  # Convert to tensor (0, 1)
            transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
        ]
    )

    def transform(examples):    
        examples = [preprocess(image) for image in examples["image"]]    
        return {"images": examples}
        
    dataset.set_transform(transform)
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ## 对用于训练的数据可视化
    # batch = next(iter(train_dataloader))
    # rows,cols = 2,4
    # fig,axes = plt.subplots(rows,cols)
    # axes = axes.flatten()

    # for i, axis in enumerate(axes):
    #     if i < len(batch["images"]):  # 确保不超过batch大小
    #         # 将(C, H, W)转置为(H, W, C)
    #         img = batch["images"][i].permute(1, 2, 0) * 0.5 + 0.5
    #         # 确保像素值在[0,1]范围内（处理浮点数图像）
    #         img = torch.clamp(img, 0, 1).cpu().numpy()
    #         axis.imshow(img)
    #         axis.axis("off")

    # plt.savefig('/data/tyh/ws/diffusion_from_diffusers/img/02_0_datasetViz.png')

    from diffusers import DDPMScheduler# We'll learn about beta_start and beta_end in the next sections
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)


    from diffusers import UNet2DModel
    model = UNet2DModel(
        in_channels=3,  # 3 channels for RGB images
        sample_size=64,  # Specify our input size
        # The number of channels per block affects the model size
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    from torch.nn import functional as F
    num_epochs = 50  # How many runs through the data should we do?
    lr = 1e-4  # What learning rate should we use
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []  # Somewhere to store the loss values for later plotting
    # Train the model (this takes a while)
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_dataloader):
            # Load the input images
            clean_images = batch["images"].to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(device)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=device,
            ).long()

            # Add noise to the clean images according
            # to the noise magnitude at each timestep
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            # Get the model prediction for the noise
            # The model also uses the timestep as an input
            # for additional conditioning
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            # Compare the prediction with the actual noise
            loss = F.mse_loss(noise_pred, noise)
            # Store the loss for later plotting
            losses.append(loss.item())
            # Update the model parameters with the optimizer based on this loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 保存模型
    torch.save(model, f'/data/tyh/ws/diffusion_from_diffusers/weight/02_0_model_{epoch}.pth')

    # 保存训练损失
    plt.subplots(1, 2, figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training loss")
    plt.xlabel("Training step")
    plt.subplot(1, 2, 2)
    plt.plot(range(400, len(losses)), losses[400:])
    plt.title("Training loss from step 400")
    plt.xlabel("Training step")
    plt.savefig('/data/tyh/ws/diffusion_from_diffusers/img/02_0_trainingLoss.png')

    # sample 
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler)
    ims = pipeline(batch_size=4).images

    rows,cols = 1,4
    fig,axes = plt.subplots(rows,cols)
    axes = axes.flatten()

    for i,img in enumerate(ims):
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.savefig('/data/tyh/ws/diffusion_from_diffusers/img/02_0_sampleViz.png')