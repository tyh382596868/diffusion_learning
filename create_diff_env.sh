conda create -n diff python==3.9 -y

# 激活 conda 环境
. /data/tyh/miniconda/bin/activate
conda activate diff

which pip

# pip install datasets diffusers genaibook torch torchvision tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install diffusers torch torchvision tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证GPU是否可用
echo "验证PyTorch GPU支持："
python -c "import torch; print('GPU可用:', torch.cuda.is_available()); print('GPU数量:', torch.cuda.device_count()); print('当前GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))"    