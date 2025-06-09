##################################Preparing the Data####################################
exp = '03_2'

import matplotlib as mpl
from datasets import load_dataset
import pdb
mpl.rcParams["image.cmap"] = "gray_r"
root_path = '/ailab/user/tangyuhang/tyh/learn/diffusion_learning'
# pdb.set_trace()
fashion_mnist = load_dataset("fashion_mnist")
clothes = fashion_mnist["train"]["image"][:8] #List [PIL.PngImagePlugin.PngImageFile image mode=L size=28x28]:
classes = fashion_mnist["train"]["label"][:8] #List [Int]


import matplotlib.pyplot as plt 

fig,axes = plt.subplots(2,4)
axes = axes.flatten()
for i,ax in enumerate(axes):
    ax.imshow(fashion_mnist["train"]["image"][i])
    ax.set_title(fashion_mnist["train"]["label"][i])
    ax.axis('off')

# 自动调整布局
plt.tight_layout()
plt.savefig(f'{root_path}/img/fashion_mnist.png')

def show_8ims(path,ims,labels=None):
    fig,axes = plt.subplots(2,4)
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        ax.imshow(ims[i])
        # ax.set_title(labels[i])
        ax.axis('off')    


    # 自动调整布局
    plt.tight_layout()
    plt.savefig(path)

import torch
from torchvision import transforms
preprocess = transforms.Compose(    
    [        
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)        
        transforms.ToTensor(),  # Convert to tensor (0, 1)        
        transforms.Pad(2),  # Add 2 pixels on all sides        
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)    
    ]
)

def transform(examples): 
    images = [preprocess(image) for image in examples["image"]]
    return {"images": images, "labels": examples["label"]}

train_dataset = fashion_mnist["train"].with_transform(transform)

train_dataloader = torch.utils.data.DataLoader(    
    train_dataset, batch_size=256, shuffle=True
)

##################################Creating a Class-Conditioned Model####################################


from diffusers import UNet2DModel
model = UNet2DModel(
    in_channels=1,  # 1 channel for grayscale images
    out_channels=1,
    sample_size=32,
    block_out_channels=(32, 64, 128, 256),
    num_class_embeds=10,  # Enable class conditioning
)

x = torch.randn((1, 1, 32, 32))
with torch.inference_mode():    
    out = model(x, timestep=7, class_labels=torch.tensor([2])).sample

print(f'shape of model output :{out.shape}')

print(model)

##################################Training the Model####################################
from diffusers import DDPMScheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.001, beta_end=0.02
)
timesteps = torch.linspace(0, 999, 8).long()
batch = next(iter(train_dataloader))
# We load 8 images from the dataset and
# add increasing amounts of noise to them
x = batch["images"][0].expand([8, 1, 32, 32])
noise = torch.rand_like(x)
noised_x = scheduler.add_noise(x, noise, timesteps)
# pdb.set_trace()

noised_x_show = (noised_x * 0.5 + 0.5).clip(0, 1)
# 转换为 numpy 数组并调整通道顺序 (BCHW → BHWC)
noised_x_show = noised_x_show.detach().cpu().numpy()
noised_x_show = noised_x_show.transpose(0, 2, 3, 1)  # [8, 32, 32, 1]
noised_x_show = noised_x_show.squeeze(3)  # 变为 [8, 32, 32]
noised_path = f'{root_path}/img/fashion_mnist_noised_x.png'
show_8ims(noised_path,noised_x_show)

def get_device():
    """
    获取当前可用的计算设备，优先使用GPU
    
    返回:
        torch.device: 可用的计算设备
    """
    if torch.cuda.is_available():
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个可用GPU")
        return torch.device("cuda")

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # 针对Apple Silicon芯片的MPS后端
        return torch.device("mps")
    else:
        return torch.device("cpu")


from torch.nn import functional as F
from tqdm import tqdm

# Initialize the scheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02
)
num_epochs = 25
lr = 3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
losses = []  # To store loss values for plotting
device = get_device()
model = model.to(device)
save_intervals = 5

from accelerate import Accelerator
# 初始化 accelerator
accelerator = Accelerator()

# 🚩 包装模型、优化器和数据加载器
model, optimizer, dataloader = accelerator.prepare(model, optimizer, train_dataloader)


# Train the model (this takes a while!)
for epoch in (progress := tqdm(range(num_epochs))):
    for step, batch in (
        inner := tqdm(
            enumerate(train_dataloader),
            position=0,
            leave=True,
            total=len(train_dataloader),
        )
    ):
        # Load the input images and classes
        clean_images = batch["images"].to(device)
        class_labels = batch["labels"].to(device)
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
        # Note the use of class_labels
        noise_pred = model(
            noisy_images,
            timesteps,
            class_labels=class_labels,
            return_dict=False,
        )[0]
        # Compare the prediction with the actual noise
        loss = F.mse_loss(noise_pred, noise)
        # Update loss display
        inner.set_postfix(loss=f"{loss.cpu().item():.3f}")
        # Store the loss for later plotting
        losses.append(loss.item())
        # Backward pass and optimization
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        # break

    # 仅在主进程打印日志
    if accelerator.is_main_process:
        print(f"Epoch {epoch} completed")

    if (epoch+1)%5==0:

        torch.save(accelerator.unwrap_model(model).state_dict(), f'{root_path}/weight/{exp}_model_{epoch}.pth')

        # 保存训练损失
        plt.figure()
        plt.plot(losses)
        plt.title("Training loss")
        plt.xlabel("Training step")
        plt.savefig(f'{root_path}/img/{exp}_trainingLoss.png')
    # break


