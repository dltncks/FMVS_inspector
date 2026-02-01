# Training Pipeline - prepare_dl/

Scripts for preparing training data and training the U-Net crack segmentation model.

## Overview

The training pipeline consists of 4 steps:
1. **Collect frames** from video → `collect_dl_dataset.py`
2. **Label cracks** in CVAT Cloud → Export masks
3. **Prepare dataset** splits → `prep_dataset.py`
4. **Train U-Net** model → `train_unet.py`

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-torch-cu121.txt  # For GPU training
```

## Step 1: Collect Training Frames

Use `collect_dl_dataset.py` to extract frames from your inspection video.

### Configuration

Edit the script before running:

```python
# prepare_dl/collect_dl_dataset.py

# Set paths (or use environment variables)
VIDEO_PATH = os.getenv("FMVS_VIDEO_PATH", "./videos/sample.mp4")
DATASET_DIR = os.getenv("FMVS_DATASET_DIR", "./data/dl_dataset")
START_SEC = 0  # Start time in video
```

### Run Collection

```bash
# Windows
set FMVS_VIDEO_PATH=C:\path\to\video.mp4
set FMVS_DATASET_DIR=C:\path\to\dataset
python -m prepare_dl.collect_dl_dataset

# Linux
export FMVS_VIDEO_PATH=/path/to/video.mp4
export FMVS_DATASET_DIR=/path/to/dataset
python -m prepare_dl.collect_dl_dataset
```

### Interactive Controls

- `s`: Save current frame
- `SPACE`: Pause/resume playback
- `q`: Quit

### Output Structure

```
data/dl_dataset/
  <video_name>/
    images/          # Grayscale ROI crops
    roi_masks/       # ROI polygon masks
    raw_gray/        # Unmasked crops (optional)
    previews/        # BGR preview with outlines
    meta/            # JSON metadata
```

**Tip:** Collect diverse samples including:
- Clear cracks (positive samples)
- Similar-looking non-cracks (negative samples)
- Different lighting conditions
- Various electrode positions

## Step 2: Label Cracks in CVAT

### Upload to CVAT Cloud

1. Go to [cvat.ai](https://cvat.ai)
2. Create new project: "Crack Segmentation"
3. Add label: `crack` (polygon/polyline type)
4. Upload images from `dataset/<video>/images/`

### Labeling Guidelines

- **Use polygon tool** to trace crack boundaries precisely
- **Label all visible cracks** in each image
- **Create separate polygons** for disconnected cracks
- **Ignore reflections** that aren't actual cracks
- **Quality over quantity**: 100 well-labeled images > 1000 poor labels

### Export Masks

1. In CVAT: Tasks → Export task dataset
2. Format: **Segmentation mask 1.1** or **COCO 1.0**
3. Download and extract

### Organize Exported Masks

```bash
# Create training directory
mkdir -p data/dl_train/images
mkdir -p data/dl_train/masks

# Copy images from collection
cp data/dl_dataset/<video>/images/*.png data/dl_train/images/

# Copy exported masks from CVAT
cp /path/to/cvat/export/masks/*.png data/dl_train/masks/
```

## Step 3: Prepare Dataset Splits

Use `prep_dataset.py` to normalize masks and create train/val splits.

### Configuration

Edit the script or use environment variables:

```python
# prepare_dl/prep_dataset.py
ROOT = Path(os.getenv("FMVS_TRAIN_DATA_DIR", "./data/dl_train"))
VAL_RATIO = 0.10  # 10% for validation
MIN_VAL_POS = 10  # Minimum positive samples in validation
```

### Run Preparation

```bash
# Windows
set FMVS_TRAIN_DATA_DIR=C:\path\to\dl_train
python -m prepare_dl.prep_dataset

# Linux
export FMVS_TRAIN_DATA_DIR=/path/to/dl_train
python -m prepare_dl.prep_dataset
```

### What It Does

1. **Creates missing masks**: Empty masks for unlabeled images
2. **Normalizes masks**: Converts all masks to binary {0, 255}
3. **Creates splits**: Stratified train/val split preserving positive/negative ratio

### Output

```
data/dl_train/
  images/          # Grayscale training images
  masks/           # Binary masks (0=background, 255=crack)
  splits/
    train.txt      # Training filenames
    val.txt        # Validation filenames
```

### Verify Splits

```bash
# Check split distribution
python -c "
from pathlib import Path
train = Path('data/dl_train/splits/train.txt').read_text().splitlines()
val = Path('data/dl_train/splits/val.txt').read_text().splitlines()
print(f'Train: {len(train)} images')
print(f'Val:   {len(val)} images')
"
```

## Step 4: Train U-Net Model

Use `train_unet.py` to train the crack segmentation model.

### Configuration

Edit the script:

```python
# prepare_dl/train_unet.py

# Paths
ROOT = Path(os.getenv("FMVS_TRAIN_DATA_DIR", "./data/dl_train"))
OUT_DIR = Path(os.getenv("FMVS_MODEL_OUTPUT_DIR", "./dl_models"))

# Hyperparameters
INPUT_SIZE = 512    # Input image size
BATCH_SIZE = 16     # Reduce if OOM (Out Of Memory)
EPOCHS = 30         # Training epochs
LR = 1e-3          # Learning rate

# Hardware
NUM_WORKERS = 4     # Data loading workers (Windows: 0-8)
USE_AMP = True      # Mixed precision (faster on RTX GPUs)

# Augmentation
AUG_FLIP = True
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
AUG_BLUR_PROB = 0.10
```

### Run Training

```bash
# Windows with GPU
set FMVS_TRAIN_DATA_DIR=C:\path\to\dl_train
set FMVS_MODEL_OUTPUT_DIR=C:\path\to\dl_models
python -m prepare_dl.train_unet

# Linux with GPU
export FMVS_TRAIN_DATA_DIR=/path/to/dl_train
export FMVS_MODEL_OUTPUT_DIR=/path/to/dl_models
python -m prepare_dl.train_unet
```

### Training Output

```
torch: 2.1.2+cu121
cuda : True
gpu  : NVIDIA GeForce RTX 4090
device: cuda

Epoch 01/30 | tr_loss=0.4523 va_loss=0.3876 va_dice=0.6234 va_iou=0.5123 time=45.2s
  [SAVE] best updated: val_dice=0.6234
Epoch 02/30 | tr_loss=0.3234 va_loss=0.2987 va_dice=0.7012 va_iou=0.6234 time=44.8s
  [SAVE] best updated: val_dice=0.7012
...
[DONE] Training complete.
```

### Output Model

```
dl_models/
  unet_crack_best.pt  # Best model checkpoint
```

### Monitor Training

**Key metrics:**
- `tr_loss`: Training loss (should decrease)
- `va_loss`: Validation loss (should decrease)
- `va_dice`: Dice coefficient (0-1, higher is better)
- `va_iou`: Intersection over Union (0-1, higher is better)

**Good results:**
- `va_dice > 0.70`: Good segmentation
- `va_iou > 0.60`: Good overlap

**If training stalls:**
- Reduce learning rate: `LR = 5e-4`
- Increase epochs: `EPOCHS = 50`
- Add more augmentation
- Collect more diverse training data

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 8  # or 4, 2

# Reduce input size
INPUT_SIZE = 256  # or 384

# Disable mixed precision
USE_AMP = False
```

### No Positive Samples

**Error:** `RuntimeError: No positive samples in train split`

**Solution:**
- Check masks are properly exported from CVAT
- Verify masks are binary (not RGB)
- Re-run `prep_dataset.py` after adding labeled masks

### Data Loading Slow (Windows)

**Issue:** Training slow on Windows

**Solution:**
```python
# Set workers to 0 on Windows
NUM_WORKERS = 0

# Or use 2-4 workers with persistent_workers
NUM_WORKERS = 2
```

### Model Not Improving

**Issue:** Validation metrics plateau

**Solutions:**
1. **Check data quality**
   - Verify mask labels are accurate
   - Ensure balanced positive/negative samples
   
2. **Increase model capacity**
   ```python
   # In UNet class instantiation
   model = UNet(in_ch=1, out_ch=1, base=64)  # Larger base
   ```

3. **Add augmentation**
   ```python
   AUG_ROTATION = True  # Add rotation augmentation
   AUG_ELASTIC = True   # Add elastic deformation
   ```

4. **Collect more data**
   - Aim for 500+ training images
   - Include diverse scenarios

## Best Practices

### Data Collection

- **Diversity:** Collect from multiple videos/conditions
- **Balance:** ~30% positive samples, ~70% negative
- **Quality:** Clear, in-focus frames
- **Quantity:** Minimum 200 images, ideally 500+

### Labeling

- **Consistency:** Use same labeling criteria throughout
- **Precision:** Trace crack boundaries accurately
- **Coverage:** Label all visible cracks in each image
- **Validation:** Have someone else review labels

### Training

- **Start small:** Test with small dataset first
- **Monitor metrics:** Watch for overfitting (train loss ↓, val loss ↑)
- **Save checkpoints:** Best model is auto-saved to `dl_models/`
- **Experiment:** Try different hyperparameters

## Advanced Topics

### Transfer Learning

Use a pretrained encoder for better results with limited data:

```python
# In train_unet.py, modify model initialization
import torchvision.models as models

class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet encoder
        resnet = models.resnet34(pretrained=True)
        # ... implement U-Net decoder
```

### Custom Augmentation

Add rotation and elastic deformation:

```python
def augment_advanced(img, mask):
    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Existing augmentations
    img, mask = augment(img, mask)
    return img, mask
```

### Multi-GPU Training

Use DataParallel for multiple GPUs:

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

## Additional Scripts

### make_empty_masks.py

Creates empty masks for all images without masks:

```bash
python -m prepare_dl.make_empty_masks
```

Use this if you have many unlabeled images and want placeholder masks.

## Summary

**Complete workflow:**

```bash
# 1. Collect frames
python -m prepare_dl.collect_dl_dataset

# 2. Label in CVAT (manual step)
# ... label cracks, export masks ...

# 3. Prepare dataset
python -m prepare_dl.prep_dataset

# 4. Train model
python -m prepare_dl.train_unet

# 5. Use trained model
# Copy dl_models/unet_crack_best.pt to main repo
# Configure config_local.py to use it
python -m fmvs_inspector
```

**Expected timeline:**
- Frame collection: 30 minutes - 2 hours
- Labeling: 2-8 hours (depends on dataset size)
- Dataset prep: 2-5 minutes
- Training: 1-4 hours (depends on GPU and dataset size)

## Questions?

See main [README.md](../README.md) for general usage or open an issue on GitHub.
