# Document Reconstruction Project

## Overview

Your task is to build a deep learning model that can **geometrically dewarp** document images captured from various angles and distances.

**Input:** RGB images of warped documents with backgrounds
**Output:** Geometrically corrected (dewarped) document image

## Important: Focus on Geometry, Not Lighting

**Key Insight:** The 3D rendered images include realistic lighting effects (shading, shadows, specular highlights). Even with a perfect geometric correction, the lighting will not match the original flat paper texture. This is expected and acceptable!

**Your goal is to:**
- ✅ Learn the correct geometric transformation (UV mapping / flow field)
- ✅ Dewarp the document to remove perspective distortion and surface warping
- ❌ NOT to remove lighting effects or match pixel-perfect intensity values

**Evaluation should focus on:**
1. **Geometric accuracy** - Are the document boundaries straight? Is text aligned?
2. **UV/flow field quality** - Does the predicted transformation make geometric sense?
3. **Visual plausibility** - Does the output look like a front-facing flat document?

**What this means for your model:**
- Predicting UV maps or flow fields is the RIGHT approach (this is geometric)
- Don't worry if pixel intensities don't exactly match ground truth due to lighting
- Focus on structural similarity (SSIM) rather than pixel-wise losses (MSE)
- The border mask helps focus on document geometry, not lighting matching

## Getting Started

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for advanced features
pip install transformers timm  # For pretrained models
pip install scikit-image       # For metrics (PSNR, SSIM)
```

### 2. Understand the Dataset

The dataset is located in `renders/synthetic_data_pitch_sweep/` with the following structure:

```
renders/synthetic_data_pitch_sweep/
├── rgb/           # Input: Warped document images (JPEG, 1080x1280)
├── ground_truth/  # Target: Flat paper textures (PNG, 512x512)
├── uv/            # UV coordinate maps (PNG, R=U, G=V, both linear in [0,1])
├── border/        # Document boundary masks (Freestyle silhouette, PNG)
└── depth/         # (Optional) Depth maps as multilayer OPEN_EXR
```

Each sample has a filename like:
```
Gemini_Generated_Image_ABCDEF_paper_paper_01_P5_R0_A04.jpg
```

**Notes about the data:**

- The **UV map** is the ground-truth coordinate map: pixel `(x, y)` in the rendered image came from texture coordinate `(u, v) = uv_map[y, x][:2]` on the flat paper, both in `[0, 1]`. The blue channel is unused (always 0). Background pixels are encoded as a uniform gray (`R == G == B`) — this is automatically used by `DocumentDataset` to derive a clean foreground mask (`sample['uv_mask']`).
- The **ground truth** has been resized to `512 × 512` to match the realistic upper bound a UV-based dewarp can reach from a `1080 × 1280` render, so reconstruction losses compare like-with-like.
- A **reference dewarping script `uv_dewarp.py`** is provided. It uses the GT UV map to splat each source pixel onto a flat canvas via bilinear forward warp — giving the *upper-bound* of what any UV-predicting model could achieve from a particular sample. Useful as a sanity check, a teaching aid, and a baseline for your model's results. Run with `python uv_dewarp.py --rgb <path> --uv <path> --side-by-side panel.png`.

### 3. Load and Visualize Data

```python
from dataset_loader import get_dataloaders, visualize_batch, DocumentDataset, visualize_sample

# Create dataloaders
train_loader, val_loader = get_dataloaders(
    data_dir='renders/synthetic_data_pitch_sweep',
    batch_size=8,
    train_split=0.8,
    img_size=(512, 512),
    use_uv=True,      # also load the GT UV map (sample['uv'])
    use_border=True,  # also load the Freestyle border (sample['border'])
)

# Quick batch overview (RGB | GT pairs)
sample_batch = next(iter(train_loader))
visualize_batch(sample_batch, num_samples=4)

# Full single-sample overview with all modalities labelled — handy for
# debugging or for figures in your report.
ds = DocumentDataset(data_dir='renders/synthetic_data_pitch_sweep',
                     use_uv=True, use_border=True)
visualize_sample(ds[0], save_path='dataset_sample.png', show=True)
```

When `use_uv=True`, every sample also carries:
- `sample['uv']` shape `[2, H, W]` in `[0, 1]` — channels are `(U, V)`. **Note: it is 2 channels, not 3** (we drop the redundant zero blue channel during loading).
- `sample['uv_mask']` shape `[1, H, W]` — pixel-perfect foreground mask auto-derived from the UV map's gray-pixel background. **Generally cleaner than `sample['border']`**, which can have disconnected Freestyle artefacts; prefer `uv_mask` for masked losses unless you have a specific reason not to.

### 4. Implement Your Model

The starter code provides a simple baseline in `dataset_loader.py`. You should improve upon this!

**Your tasks:**
1. Design a better architecture (U-Net, encoder-decoder, transformer-based, etc.)
2. Use pretrained backbones from HuggingFace or timm
3. Experiment with different loss functions
4. Add data augmentation if needed
5. Implement evaluation metrics (PSNR, SSIM, perceptual loss)

### 5. Using Pretrained Models from HuggingFace

Here's how to incorporate a pretrained backbone:

```python
import torch.nn as nn
from transformers import AutoModel

class MyDocumentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained encoder (e.g., ResNet50)
        self.encoder = AutoModel.from_pretrained("microsoft/resnet-50")

        # Add your decoder
        self.decoder = nn.Sequential(
            # Your decoder layers here
        )

    def forward(self, x):
        # Extract features from encoder
        features = self.encoder(x).last_hidden_state

        # Decode to reconstructed image
        output = self.decoder(features)
        return output
```

Or use timm for vision models:

```python
import timm

class MyDocumentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNet
        self.encoder = timm.create_model('efficientnet_b0',
                                         pretrained=True,
                                         features_only=True)

        # Your decoder
        self.decoder = ...
```

## Project Milestones

### Milestone 1: Baseline Model
- [ ] Successfully load and visualize the dataset
- [ ] Implement a simple encoder-decoder model
- [ ] Train on the dataset with MSE loss
- [ ] Evaluate on validation set

### Milestone 2: Improved Architecture
- [ ] Integrate a pretrained backbone (ResNet, EfficientNet, etc.)
- [ ] Add skip connections (U-Net style)
- [ ] Experiment with different loss functions (L1, perceptual loss, SSIM)
- [ ] Implement proper evaluation metrics

### Milestone 3: Advanced Techniques (Optional)
- [ ] Use depth/UV information if available
- [ ] Add attention mechanisms
- [ ] Implement adversarial training (GAN)
- [ ] Try transformer-based architectures
- [ ] Ensemble multiple models

## Tips and Suggestions

### Architecture Ideas
1. **U-Net with Pretrained Encoder**: Use ResNet/EfficientNet as encoder, add skip connections
2. **Attention U-Net**: Add attention gates to focus on document regions
3. **Transformer-based**: Use vision transformers (ViT, Swin) for global context
4. **Multi-scale**: Process images at multiple resolutions
5. **Spatial Transformer Network (STN)**: Predict flow fields and use `grid_sample` for differentiable warping (see below!)

### Using torch.nn.functional.grid_sample for Geometric Warping

**Key Insight**: Document reconstruction is fundamentally a geometric transformation problem. Instead of just predicting pixel values, you can predict a deformation field that warps the distorted document back to its flat form.

`torch.nn.functional.grid_sample` allows you to perform **differentiable image warping**, meaning you can:
1. Predict a flow/deformation field with a neural network
2. Warp the input image according to this field
3. Backpropagate through the warping operation

**Example Architecture with grid_sample:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_loader import create_base_grid

class FlowBasedDocumentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder predicts a flow field
        self.encoder = ... # Your encoder (ResNet, U-Net, etc.)

        # Flow prediction head: predicts [B, 2, H, W]
        # representing (x, y) displacement at each pixel
        self.flow_head = nn.Conv2d(features_dim, 2, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract features
        features = self.encoder(x)

        # Predict flow field [B, 2, H, W]
        flow = self.flow_head(features)

        # Create base sampling grid in range [-1, 1]
        base_grid = create_base_grid(B, H, W, x.device)  # [B, H, W, 2]

        # Add predicted flow to base grid
        # Need to permute flow from [B, 2, H, W] to [B, H, W, 2]
        sampling_grid = base_grid + flow.permute(0, 2, 3, 1)

        # Warp the input image using the predicted grid
        reconstructed = F.grid_sample(
            x,
            sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return reconstructed
```

**Why this works:**
- The network learns to predict pixel correspondences between warped and flat document
- `grid_sample` performs bilinear interpolation (smooth, differentiable)
- The warping is end-to-end trainable with standard losses

**Advanced variant:**
You can also predict both a flow field AND a refined texture:
```python
warped = F.grid_sample(x, sampling_grid, ...)
refinement = self.refinement_net(warped)
output = warped + refinement  # Coarse-to-fine approach
```

### Loss Functions

**IMPORTANT:** Since the goal is geometric correction (not photometric matching), choose loss functions that emphasize structure over pixel intensities.

#### Recommended: Structural Losses

**SSIM Loss (Structural Similarity) - RECOMMENDED**
```python
from pytorch_msssim import ssim, SSIM

# During training
ssim_loss = 1 - ssim(predicted, ground_truth, data_range=1.0)
```
- Measures structural similarity (edges, patterns) rather than pixel values
- **Perfect for geometric reconstruction** - ignores lighting differences
- More perceptually aligned than L1/L2
- Install: `pip install pytorch-msssim`

**Perceptual Loss (VGG-based)**
```python
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        return F.mse_loss(pred_features, target_features)
```
- Compares high-level features, not raw pixels
- Robust to lighting variations
- Focuses on structure and content

#### Basic Reconstruction Losses (Use with caution)

**L1 Loss (Mean Absolute Error)**
```python
criterion = nn.L1Loss()
loss = criterion(predicted, ground_truth)
```
- Simple baseline
- Will try to match pixels exactly (including lighting)
- May penalize geometrically correct but differently lit outputs

**MSE Loss (L2 Loss)**
```python
criterion = nn.MSELoss()
loss = criterion(predicted, ground_truth)
```
- Standard choice but very sensitive to lighting
- Heavily penalizes brightness differences
- **Not recommended as primary loss** for this task

#### Advanced: Masked Losses (IMPORTANT!)

**Why use masking?** Your dataset includes background pixels that are not part of the document. Without masking, the network wastes capacity learning to reconstruct backgrounds instead of focusing on the document surface.

The dataset provides two ways to get a foreground mask:

- **`sample['uv_mask']` (recommended)** — auto-derived from the UV map's gray-pixel background. Pixel-perfect, no holes, available whenever you set `use_uv=True`. No separate file needed.
- **`sample['border']`** — a Freestyle silhouette rendered as PNG in `border/`. Available when `use_border=True`, but can have disconnected edge artefacts.

```python
from dataset_loader import MaskedL1Loss

# Load data with UV maps (gives us both `uv` and `uv_mask`)
train_loader, val_loader = get_dataloaders(
    data_dir='renders/synthetic_data_pitch_sweep',
    use_uv=True,      # also exposes sample['uv_mask']
    batch_size=8,
)

# Use masked loss
criterion = MaskedL1Loss(use_mask=True)

# In training loop
for batch in train_loader:
    rgb = batch['rgb'].to(device)
    gt = batch['ground_truth'].to(device)
    mask = batch['uv_mask'].to(device)  # [B, 1, H, W] — clean, from UV map

    output = model(rgb)
    loss = criterion(output, gt, mask)  # Only compute loss on document pixels
```

**Benefits of masking:**
- Network focuses on document reconstruction, not background
- Faster convergence
- Better quality on document regions
- Prevents "cheating" by predicting backgrounds

#### UV Map Supervision (Advanced)

If your model explicitly predicts UV coordinates (as in the flow-based approach), you can supervise the UV prediction directly. Two important things to know up front:

1. **`batch['uv']` has shape `[B, 2, H, W]`**, not `[B, 3, H, W]` — channels are `(U, V)`, both already in `[0, 1]` (no ImageNet normalisation). Your model's `pred_uv` head should output 2 channels too, in the same range.
2. **Use `loss_type='l1'` or `'mse'` for UV supervision, not `'ssim'`.** SSIM expects 3-channel images and will fail on 2-channel UV tensors. (The image branch can still use SSIM — that's separate.)

```python
from dataset_loader import UVReconstructionLoss

# Load with UV maps (also exposes batch['uv_mask'])
train_loader, val_loader = get_dataloaders(
    data_dir='renders/synthetic_data_pitch_sweep',
    use_uv=True,
    batch_size=8,
)

# Combined loss — l1/mse works on both image and UV.
criterion = UVReconstructionLoss(
    reconstruction_weight=1.0,  # Weight for final image
    uv_weight=0.5,              # Weight for UV map
    smoothness_weight=0.01,     # Weight for flow smoothness (TV)
    use_mask=True,
    loss_type='l1',             # 'l1' or 'mse' for the UV branch
)

# In training
for batch in train_loader:
    rgb = batch['rgb'].to(device)
    gt = batch['ground_truth'].to(device)
    gt_uv = batch['uv'].to(device)         # [B, 2, H, W] in [0, 1]
    mask = batch['uv_mask'].to(device)     # [B, 1, H, W] from UV map

    # Model predicts both image and UV (2-channel UV head!)
    output, predicted_uv, flow = model(rgb)

    losses = criterion(
        pred_image=output,
        target_image=gt,
        pred_uv=predicted_uv,
        target_uv=gt_uv,
        flow=flow,
        mask=mask,
    )

    total_loss = losses['total']
    total_loss.backward()
```

#### Other Advanced Losses

- **Perceptual Loss**: Use VGG features for better perceptual quality
- **SSIM Loss**: Structural similarity metric
- **Adversarial Loss**: Add a discriminator for sharper results (GANs)
- **Flow Smoothness (TV Loss)**: Regularize flow fields to be spatially smooth (included in `UVReconstructionLoss`)

### Training Tips
1. Start with lower resolution (256x256) for faster experimentation
2. Use learning rate scheduling (reduce on plateau)
3. Monitor both train and validation loss
4. Visualize predictions during training
5. Save checkpoints regularly
6. Use gradient clipping if training is unstable

### Data Augmentation
Since the dataset is synthetic, you might want to add augmentation:
- Random crops
- Color jittering
- Gaussian noise
- Random flips (be careful with document orientation!)

## Evaluation Metrics

**Important:** Since the focus is on geometric correction (not photometric matching), prioritize metrics that measure structural quality over pixel-wise accuracy.

### Primary Metrics (Geometric Quality)

**1. SSIM (Structural Similarity Index) - MOST IMPORTANT**
```python
from skimage.metrics import structural_similarity

def compute_ssim(pred, target):
    """
    Compute SSIM - measures structural similarity.
    Higher is better (range 0-1, where 1 is perfect).
    Target: SSIM > 0.8 for good geometric reconstruction.
    """
    # Convert to numpy [H, W, C]
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)

    ssim_score = structural_similarity(
        target_np,
        pred_np,
        channel_axis=2,
        data_range=1.0
    )
    return ssim_score
```

**2. Geometric Error Metrics**
```python
def compute_corner_error(pred_flow, gt_flow, corners):
    """
    Measure error at document corners.
    Tests if the model correctly maps corner positions.
    """
    pred_corners = warp_points(corners, pred_flow)
    gt_corners = warp_points(corners, gt_flow)
    error = np.mean(np.linalg.norm(pred_corners - gt_corners, axis=1))
    return error

def compute_line_straightness(dewarped_image):
    """
    Measure how straight text lines are after dewarping.
    Straight lines = good geometric correction.
    """
    # Use edge detection to find text lines
    edges = cv2.Canny(dewarped_image)
    lines = cv2.HoughLines(edges, ...)
    # Measure deviation from horizontal/vertical
    straightness_score = ...
    return straightness_score
```

### Secondary Metrics (Photometric Quality)

**PSNR (Peak Signal-to-Noise Ratio)**
```python
from skimage.metrics import peak_signal_noise_ratio

def compute_psnr(pred, target):
    """
    Compute PSNR - measures pixel-wise accuracy.
    NOTE: Will be affected by lighting differences!
    Use as secondary metric only.
    """
    pred_np = pred.cpu().numpy().transpose(1, 2, 0)
    target_np = target.cpu().numpy().transpose(1, 2, 0)

    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    return psnr
```

**Note on PSNR:** Because of lighting differences between warped and ground truth images, PSNR may be lower even with perfect geometric correction. This is expected! Don't optimize solely for PSNR.

### Visual Quality Assessment (Qualitative)

The most important evaluation is visual inspection:

```python
def visualize_results(input_img, prediction, ground_truth, flow_field=None):
    """
    Create visualization comparing geometric correction quality.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Input (Warped)')

    axes[0, 1].imshow(prediction)
    axes[0, 1].set_title('Prediction (Dewarped)')

    axes[0, 2].imshow(ground_truth)
    axes[0, 2].set_title('Ground Truth')

    # Row 2: Analysis
    # Difference map (structural, not photometric)
    axes[1, 0].imshow(np.abs(sobel(prediction) - sobel(ground_truth)))
    axes[1, 0].set_title('Edge Difference (Geometric)')

    # Flow field visualization (if available)
    if flow_field is not None:
        axes[1, 1].imshow(flow_to_color(flow_field))
        axes[1, 1].set_title('Predicted Flow Field')

    # Grid overlay showing distortion correction
    axes[1, 2].imshow(overlay_grid(prediction))
    axes[1, 2].set_title('Grid Overlay (Check Straightness)')

    plt.tight_layout()
    plt.show()
```

### Recommended Evaluation Strategy

1. **Training**: Optimize for SSIM loss (+ flow smoothness if using flow-based model)
2. **Validation**: Report SSIM (primary) and PSNR (secondary)
3. **Final Evaluation**:
   - Quantitative: SSIM score across test set
   - Qualitative: Visual inspection of dewarped documents
   - Geometric: Check if text lines are straight, corners are correct
   - Bonus: Test on real document images (generalization)

### Expected Performance Targets

- **SSIM**: > 0.8 (good), > 0.9 (excellent)
- **PSNR**: > 20 dB (acceptable), > 25 dB (good) - but remember lighting affects this!
- **Visual Quality**: Text should be readable, lines straight, no distortion artifacts

## Example Training Script

```python
from dataset_loader import get_dataloaders, DocumentReconstructionModel
import torch
import torch.nn as nn

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader = get_dataloaders(
    data_dir='renders/synthetic_data_pitch_sweep',
    batch_size=8
)

# Model
model = DocumentReconstructionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(50):
    model.train()
    for batch in train_loader:
        rgb = batch['rgb'].to(device)
        gt = batch['ground_truth'].to(device)

        output = model(rgb)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    # ... add validation code
```

## Common Issues and Solutions

### Issue: Out of Memory
- Reduce batch size
- Reduce image resolution
- Use gradient accumulation

### Issue: Model not converging
- Check learning rate (try 1e-4, 1e-3)
- Verify data normalization
- Visualize inputs and outputs
- Try simpler architecture first

### Issue: Blurry outputs
- Add perceptual loss or SSIM loss
- Reduce weight of L2 loss
- Try adversarial training

## Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **HuggingFace Models**: https://huggingface.co/models
- **timm Models**: https://github.com/huggingface/pytorch-image-models
- **U-Net Paper**: https://arxiv.org/abs/1505.04597
- **Perceptual Loss Paper**: https://arxiv.org/abs/1603.08155

## Submission Guidelines

Your submission should include:
1. Your model implementation (`model.py`)
2. Training script (`train.py`)
3. Evaluation script (`evaluate.py`)
4. Trained model weights (`best_model.pth`)
5. Results visualization (sample predictions)
6. Brief report (1-2 pages) describing:
   - Your architecture choices
   - Loss functions used
   - Training details (hyperparameters, augmentation)
   - Quantitative results (PSNR, SSIM, loss curves)
   - Qualitative results (visual samples)
   - What worked and what didn't

## Getting Help

- Review the provided `dataset_loader.py` carefully
- Start with the simple baseline model
- Gradually add complexity
- Visualize intermediate outputs to debug
- Compare your outputs with ground truth frequently

Good luck! 🚀
