##################################Sampling####################################
exp = '03_3'
root_path = '/ailab/user/tangyuhang/tyh/learn/diffusion_learning'

import torch
from tqdm import tqdm
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
    
device = get_device()

from diffusers import DDPMScheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_start=0.001, beta_end=0.02
)

from diffusers import UNet2DModel
model = UNet2DModel(
    in_channels=1,  # 1 channel for grayscale images
    out_channels=1,
    sample_size=32,
    block_out_channels=(32, 64, 128, 256),
    num_class_embeds=10,  # Enable class conditioning
)

# 加载权重（CPU 加载）
state_dict = torch.load(
    "/ailab/user/tangyuhang/tyh/learn/diffusion_learning/weight/03_2_model_24.pth"
)
model.load_state_dict(state_dict)
model.to(device)


# 可选：切换至评估模式（如测试时）
model.eval()

def generate_from_class(class_to_generate, n_samples=8):
    sample = torch.randn(n_samples, 1, 32, 32).to(device)
    class_labels = [class_to_generate] * n_samples
    class_labels = torch.tensor(class_labels).to(device)
    for _, t in tqdm(enumerate(scheduler.timesteps)):
        # Get model prediction
        with torch.inference_mode():
            noise_pred = model(sample, t, class_labels=class_labels).sample
        # Update sample with step
        sample = scheduler.step(noise_pred, t, sample).prev_sample
    return sample.clip(-1, 1) * 0.5 + 0.5

# label = 0
for label in range(10):
    images = generate_from_class(label)
    # import pdb
    # pdb.set_trace()
    img_show = images.detach().cpu().numpy()
    img_show = img_show.transpose(0,2,3,1)
    img_show = img_show.squeeze(3)

    import matplotlib.pyplot as plt

    fig,axes = plt.subplots(2,4)
    axes = axes.flatten()

    for i,ax in enumerate(axes):
        ax.imshow(img_show[i])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{root_path}/img/{exp}_{label}.png')


