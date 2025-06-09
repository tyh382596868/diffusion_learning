from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pdb

def get_dataset(name,image_size=64):

    if name=='smithsonian_butterflies_subset':
        dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        print(dataset)
        # from torchvision import transforms
        image_size = image_size
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

        return dataset

    elif name=='pokemon':
        # load the dataset
        dataset = load_dataset('huggan/pokemon',split='train')

        # trasnform the dataset
        # from torchvision import transforms

        # We keep this higher than in the book in this part for visualization
        image_size = image_size

        # Define transformations
        preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # Resize
                transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
                transforms.ToTensor(),  # Convert to tensor (0, 1)
                transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
            ]
        )

        # Apply transformations to the dataset
        def transform(examples):
            examples = [preprocess(image) for image in examples["image"]]
            return {"images": examples}

        dataset.set_transform(transform)

        return dataset

    elif name=='mnist':
        image_size = image_size


        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # 添加调整大小操作
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]范围
        ])

        # 加载MNIST训练集
        dataset = datasets.MNIST(
            root='/data/tyh/ws/diffusion_from_scrach/data',          # 数据存储路径
            train=True,             # 使用训练集
            download=True,          # 自动下载
            transform=transform     # 应用预处理
        )

        return dataset

if __name__=="__main__":
    device = 'cuda'
    Multi_gpu = True
    amp = True

    name = 'pokemon'
    image_size = 64
    dataset = get_dataset(name,image_size=image_size)

    # 根据不同的数据集设定模型参数
    if name == 'pokemon' or name == 'smithsonian_butterflies_subset':
        in_channels=3
        out_channels=3
    elif name == 'mnist':
        in_channels=1
        out_channels=1

    batch_size = 32*4
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=4)

    # 对用于训练的数据可视化
    if name == 'pokemon' or name == 'smithsonian_butterflies_subset':
        batch = next(iter(train_dataloader))
        rows,cols = 2,4
        fig,axes = plt.subplots(rows,cols)
        axes = axes.flatten()

        for i, axis in enumerate(axes):
            if i < len(batch["images"]):  # 确保不超过batch大小
                # 将(C, H, W)转置为(H, W, C)
                img = batch["images"][i].permute(1, 2, 0) * 0.5 + 0.5
                # 确保像素值在[0,1]范围内（处理浮点数图像）
                img = torch.clamp(img, 0, 1).cpu().numpy()
                axis.imshow(img)
                axis.axis("off")

        plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_datasetViz.png')

    from diffusers import DDPMScheduler# We'll learn about beta_start and beta_end in the next sections
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)


    from diffusers import UNet2DModel
    model = UNet2DModel(
        in_channels=in_channels,  # 3 channels for RGB images
        out_channels=out_channels,
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

    if Multi_gpu:
        # 多GPU训练支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

    from torch.nn import functional as F
    num_epochs = 50  # How many runs through the data should we do?
    lr = 1e-4  # What learning rate should we use
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []  # Somewhere to store the loss values for later plotting
    # Train the model (this takes a while)
    if amp:
        scaler = GradScaler()
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(train_dataloader):
            if name == 'pokemon' or name == 'smithsonian_butterflies_subset':
                # Load the input images
                clean_images = batch["images"].to(device)
            else:

                clean_images = batch[0].to(device) #batch是list，索引0是图像，1是标签
            # pdb.set_trace()
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
            if amp:
                with autocast():
                    # Forward pass through the model
                    noise_pred = model(sample=noisy_images, timestep=timesteps, return_dict=False)[0]
                    # Compare the prediction with the actual noise
                    loss = F.mse_loss(noise_pred, noise)
            else:
                noise_pred = model(sample=noisy_images, timestep=timesteps, return_dict=False)[0]
                # Compare the prediction with the actual noise
                loss = F.mse_loss(noise_pred, noise)

            if amp:
                losses.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()      

            else:          
                # Store the loss for later plotting
                losses.append(loss.item())
                # Update the model parameters with the optimizer based on this loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # break
        if epoch % 5==0:

            # sample 
            pipeline = DDPMPipeline(unet=model.module if isinstance(model, torch.nn.DataParallel) else model, scheduler=scheduler)
            ims = pipeline(batch_size=4, output_type="pt").images

            rows,cols = 1,4
            fig,axes = plt.subplots(rows,cols)
            axes = axes.flatten()

            for i,img in enumerate(ims):
                axes[i].imshow(img)
                axes[i].axis('off')

            plt.savefig(f'/data/tyh/trash/02_0_{name}_{image_size}_{epoch}_sampleViz.png')  

    # 保存模型
    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), f'/data/tyh/ws/diffusion_from_diffusers/weight/02_0_{name}_{image_size}_model_{epoch}.pth')

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
    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_trainingLoss.png')

    # sample 
    pipeline = DDPMPipeline(unet=model.module if isinstance(model, torch.nn.DataParallel) else model, scheduler=scheduler)
    ims = pipeline(batch_size=4).images

    rows,cols = 1,4
    fig,axes = plt.subplots(rows,cols)
    axes = axes.flatten()

    for i,img in enumerate(ims):
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_0_{name}_{image_size}_sampleViz.png')