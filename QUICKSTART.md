# Quick Start Guide

Get FMVS Inspector running in 5 minutes.

## Prerequisites

- Python 3.10+
- GPU with CUDA 12.1 (optional, for DL mode)

## Installation

### Windows (GPU)

```cmd
REM 1. Clone repository
git clone https://github.com/yourusername/fmvs_inspector.git
cd fmvs_inspector

REM 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

REM 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-torch-cu121.txt

REM 4. Configure paths
copy fmvs_inspector\inspection\config_local.py.template fmvs_inspector\inspection\config_local.py
notepad fmvs_inspector\inspection\config_local.py

REM Edit these lines:
REM   video_path=r"C:\path\to\video.mp4"
REM   log_dir=r"C:\path\to\logs"
REM   img_dir=r"C:\path\to\images"

REM 5. Add trained model
REM Copy your unet_crack_best.pt to: dl_models\unet_crack_best.pt

REM 6. Run inspection
python -m fmvs_inspector
```

### Linux (GPU)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fmvs_inspector.git
cd fmvs_inspector

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-torch-cu121.txt

# 4. Configure paths
cp fmvs_inspector/inspection/config_local.py.template \
   fmvs_inspector/inspection/config_local.py
nano fmvs_inspector/inspection/config_local.py

# Edit these lines:
#   video_path="/path/to/video.mp4"
#   log_dir="/path/to/logs"
#   img_dir="/path/to/images"

# 5. Add trained model
# Copy your unet_crack_best.pt to: dl_models/unet_crack_best.pt

# 6. Run inspection
python -m fmvs_inspector
```

### CPU Only (OpenCV Mode)

Skip PyTorch installation and use OpenCV mode:

```bash
# Install only core dependencies
pip install -r requirements.txt

# In config_local.py, set:
#   mode="opencv"

# Run
python -m fmvs_inspector
```

## Usage

### ROI Selection

When the inspection starts:
1. **Left-click** to add polygon points around the electrode
2. **Right-click** when done
3. Press **'c'** to clear and start over
4. Press **'q'** to cancel

### During Inspection

- **SPACE**: Pause/resume
- **r**: Reselect ROI (when paused)
- **q**: Quit

## Outputs

Check these directories (as configured):

```
logs/<video_name>/<video_name>_detections.txt
images/<video_name>/<video_name>_HHMMSS_orig.png
images/<video_name>/<video_name>_HHMMSS_insp.png
```

## Troubleshooting

### No module named 'torch'

```bash
# Install PyTorch for GPU
pip install -r requirements-torch-cu121.txt

# Or use OpenCV mode (no PyTorch needed)
```

### CUDA not available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, either:
# 1. Install CUDA 12.1 from nvidia.com
# 2. Use CPU mode (slower)
# 3. Use OpenCV mode
```

### Video won't open

```bash
# Convert video to H.264
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4
```

### No detections found

Try lowering the threshold:

```python
# In config_local.py
dl=replace(
    BASE.dl,
    model=replace(BASE.dl.model, mask_thr=0.30),  # Lower threshold
),
```

## Next Steps

- **Full documentation**: See [README.md](README.md)
- **Training guide**: See [prepare_dl/README.md](prepare_dl/README.md)
- **Configuration options**: Edit `fmvs_inspector/inspection/config.py`

## Need Help?

- Check [README.md](README.md) for detailed docs
- Open an issue on GitHub
- Email: your.email@example.com
