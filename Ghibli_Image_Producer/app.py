# app.py (Gradio UI with blue elegant theme + sharable link)
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# -------------------------
# Config
# -------------------------
CHECKPOINT_PATH = "/teamspace/studios/this_studio/.lightning_studio/finetune_epoch_12.pth"
OUT_DIR = "/teamspace/studios/this_studio/.lightning_studio/Web_Outputs"
IMG_SIZE = 512
N_RESIDUAL = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Generator architecture
# -------------------------
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
        # Downsample
        for _ in range(2):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]
        # Upsample
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

# -----------------------
# Load generator (B→A)
# -----------------------
generator = Generator(n_residual_blocks=N_RESIDUAL).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

if not isinstance(ckpt, dict) or "G_A2B" not in ckpt:
    raise RuntimeError(
        f"Checkpoint must contain 'G_A2B'. Found keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'not a dict'}"
    )

generator.load_state_dict(ckpt["G_A2B"], strict=True)
generator.eval()
print(f"Loaded G_A2B weights from {CHECKPOINT_PATH}")

# -----------------------
# Pre/Post processing
# -----------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # [0,1]->[-1,1]
])

def tensor_to_pil(t):
    if t.dim() == 4:
        t = t[0]
    t = (t.clamp(-1, 1) + 1.0) / 2.0  # [-1,1] -> [0,1]
    return transforms.ToPILImage()(t.cpu())

# -------------------------
# Inference
# -------------------------
def infer_and_save(input_image, high_quality: bool=False):
    start = time.time()
    if input_image is None:
        return None, None, "⚠️ No input provided."
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    x = preprocess(input_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fake = generator(x)

    out_pil = tensor_to_pil(fake)
    fname = f"result_{int(time.time()*1000)}.png"
    out_path = os.path.join(OUT_DIR, fname)
    out_pil.save(out_path, format="PNG")

    elapsed = time.time() - start
    return out_pil, out_path, f"✅ Done in {elapsed:.2f}s — Saved as {fname}"

# -------------------------
# CSS (Blue elegant theme)
# -------------------------
CUSTOM_CSS = """
/* Background */
body {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    font-family: 'Inter', sans-serif;
    color: #1e3a8a;
}

/* Headings */
h1 {
    color: #0d47a1;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5em;
}

/* Subtext */
.lead {
    font-size: 1.05rem;
    color: #37474f;
    text-align: center;
    margin-bottom: 1em;
}

/* Card Styling */
.card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    padding: 18px;
    transition: transform 0.2s ease-in-out;
}
.card:hover {
    transform: translateY(-3px);
}

/* Buttons */
button {
    background: #1e88e5 !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: background 0.3s ease, transform 0.2s ease-in-out;
}
button:hover {
    background: #1565c0 !important;
    transform: scale(1.03);
}
#generate-btn {
    background: #43a047 !important; /* green accent */
}
#generate-btn:hover {
    background: #2e7d32 !important;
}
#download-btn {
    background: #f57c00 !important; /* orange accent */
}
#download-btn:hover {
    background: #e65100 !important;
}

/* Checkbox & Inputs */
input[type="checkbox"] {
    accent-color: #1e88e5;
}
.gradio-container input, .gradio-container textarea {
    border-radius: 10px !important;
    border: 1px solid #90caf9 !important;
}

/* Images */
.gr-image {
    border-radius: 12px;
    border: 2px solid #90caf9;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Footer note */
.footer-note {
    font-size: 0.9rem;
    color: #455a64;
    margin-top: 10px;
    text-align: center;
}
"""


# -------------------------
# Build Gradio UI
# -------------------------
with gr.Blocks(css=CUSTOM_CSS, title="CycleGAN A→B Demo") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 🎨 CycleGAN Demo  \n*Ghibli → Photo (A→B)*")
            gr.Markdown(f"📂 Using checkpoint: **{os.path.basename(CHECKPOINT_PATH)}**")
            gr.Markdown("Upload an image, generate output, and download the result below.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                inp = gr.Image(label="📤 Upload Ghibli-style image", type="pil")
                with gr.Row():
                    gen_btn = gr.Button("✨ Generate", elem_id="generate-btn")
                    clear_btn = gr.Button("🧹 Clear")
                highq = gr.Checkbox(label="High quality (same pass for now)", value=False)
                gr.Markdown("💡 Tip: Use clean inputs for better results.")

        with gr.Column(scale=1):
            with gr.Group():
                out_img = gr.Image(label="📸 Result (Photo-style)", type="pil")
                file_out = gr.File(label="⬇️ Download result (PNG)", interactive=False)
                status = gr.Markdown("", elem_id="status-text")
                with gr.Row():
                    dl_btn = gr.Button("Download", elem_id="download-btn")

    # Wire buttons
    gen_btn.click(fn=infer_and_save, inputs=[inp, highq], outputs=[out_img, file_out, status])
    clear_btn.click(lambda: (None, None, ""), None, [out_img, file_out, status])
    dl_btn.click(lambda path: path, inputs=file_out, outputs=file_out)

# -------------------------
# Launch with sharable link
# -------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # 🔥 sharable link enabled
    )
