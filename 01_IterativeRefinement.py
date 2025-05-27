import torch
from diffusers import DDPMPipeline


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
# Set the device to use our GPU or CPU
 
device = get_device()
print(device)

# Load the pipeline
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

# Sample an image
image_pipe().images[0]

