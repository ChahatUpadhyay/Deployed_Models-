# test_ghibli_from_ckpt.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


checkpoint_path = "/teamspace/studios/this_studio/.lightning_studio/checkpoints/cyclegan_epoch_85.pth"
input_image_path = "/teamspace/studios/this_studio/.lightning_studio/test_5.jpg"
out_dir = "/teamspace/studios/this_studio/.lightning_studio/Output_2"
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# DEVICE
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=12):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        ]
        in_channels = 64
        # Downsample x2
        for _ in range(2):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        # 12 residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]
        
        for _ in range(2):
            out_channels = in_channels // 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)


G = Generator(n_residual_blocks=12).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)

if not isinstance(ckpt, dict) or "G_B2A" not in ckpt:
    raise RuntimeError(
        "Checkpoint must be a dict containing 'G_B2A'. "
        f"Found keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'not a dict'}"
    )

G.load_state_dict(ckpt["G_B2A"], strict=True)
G.eval()
print("Loaded G_A2B weights (strict=True).")


img_size = 512  # same as training
preprocess = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # [0,1]->[-1,1]
])

def tensor_to_pil(t):
    """
    Accepts tensor (1,C,H,W) or (C,H,W) in [-1,1], returns PIL.Image.
    """
    if t.dim() == 4:
        t = t[0]
    t = (t.clamp(-1, 1) + 1.0) / 2.0  # [-1,1] -> [0,1]
    t = t.cpu()
    return transforms.ToPILImage()(t)

# -----------------------
# INFERENCE
# -----------------------
img = Image.open(input_image_path).convert("RGB")
inp = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = G(inp)

out_pil = tensor_to_pil(out)

base = os.path.splitext(os.path.basename(input_image_path))[0]
out_path = os.path.join(out_dir, f"{base}_ghibli.png")
out_pil.save(out_path)
print("Saved:", out_path)
