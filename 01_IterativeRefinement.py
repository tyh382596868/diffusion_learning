import torch
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
# Set the device to use our GPU or CPU
 
device = get_device()
print(device)

# Load the pipeline
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

# # Sample an image

# img = image_pipe().images[0]


# plt.figure()
# plt.imshow(img)
# plt.axis('off') # 关闭坐标轴
# plt.savefig('/data/tyh/ws/diffusion_from_diffusers/img/celebahq.png')



# Sample an image
batchsize = 8
imgs = image_pipe(batch_size = batchsize).images
"""使用Matplotlib创建网格图"""
rows, cols = 2,4
fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
axes = axes.flatten()

for i, img in enumerate(imgs):
    axes[i].imshow(img)
    axes[i].axis('off')  # 隐藏坐标轴

plt.tight_layout()  # 调整布局
plt.savefig('/data/tyh/ws/diffusion_from_diffusers/img/delebahqs.png', dpi=300, bbox_inches='tight')


