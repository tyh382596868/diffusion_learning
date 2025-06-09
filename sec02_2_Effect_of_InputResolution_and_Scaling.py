# 图像分辨率(image resolution eg.64*64)和图像的像素值的范围(input scaling eg.floats between 0 and 1;floats between –1 and 1)会影响图像添加噪声后的效果。
from sec02_1_Training_my_DiffusionModel import get_dataset
from diffusers import DDPMScheduler
import torch
import matplotlib.pyplot as plt

if __name__=="__main__":
    device = 'cuda'
    batch_size = 8

# 1. import different dataset    

    names = ['mnist','smithsonian_butterflies_subset','pokemon']
    name = names[2]
    image_size = 512
    dataset = get_dataset(name,image_size=image_size)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    batch = next(iter(train_dataloader))
    if name == 'mnist':
        clean_imgs = batch[0]


    elif name == 'pokemon' or name == 'smithsonian_butterflies_subset':
        clean_imgs = batch["images"]


    else:
        # 处理未知数据集的情况
        raise ValueError(f"不支持的数据集名称: {name}")

# 2. noise schedule and add noise to clean image

    # We'll learn about beta_start and beta_end in the next sections
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.02)

    # Create a tensor with 8 evenly spaced values from 0 to 999
    timesteps = torch.linspace(0, 999, 8).long()

    noise = torch.rand_like(clean_imgs)

    noised = scheduler.add_noise(clean_imgs,noise,timesteps)



# 3. save image

    fig,axes = plt.subplots(2,4)
    axes = axes.flatten()
    for i,axis in enumerate(axes):

        print(noised[i].max(),noised[i].min())
        print(noised[i].shape)

        if name == 'mnist':
            img = noised[i].squeeze()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 加小常数避免除零
            axis.imshow(img, cmap='gray')


        elif name == 'pokemon' or name == 'smithsonian_butterflies_subset':
            img = noised[i].permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # 加小常数避免除零
            axis.imshow(img)


        else:
            # 处理未知数据集的情况
            raise ValueError(f"不支持的数据集名称: {name}")

        
        axis.axis('off')

    plt.savefig(f'/data/tyh/ws/diffusion_from_diffusers/img/02_2_{name}_{image_size}.png')




