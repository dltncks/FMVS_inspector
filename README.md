# FMVS Inspector - Deep Learning Crack Detection

Automated crack detection system for rewinding electrode inspection using U-Net semantic segmentation with OpenCV fallback.

## Features

- 🔍 **Deep learning primary detection** (U-Net) with traditional CV (OpenCV) fallback
- 🎯 **Interactive ROI selection** via polygon drawing
- 📊 **Temporal voting** for robust detection across frames
- 💾 **Automatic logging** with video timestamps  
- 🖼️ **Image pairs saved** (original + annotated) for review
- 🐛 **Debug visualization** grid for algorithm tuning

## Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for DL mode)
- Windows 10/11 or Linux

### Installation

**1. Clone repository:**
```bash
git clone https://github.com/yourusername/fmvs_inspector.git
cd fmvs_inspector
```

**2. Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
# Core dependencies (required for both modes)
pip install -r requirements.txt

# GPU support (for DL mode with CUDA 12.1):
pip install -r requirements-torch-cu121.txt

# (Optional) Verify:
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**4. Download or train model:**
- **Option A:** Place your trained `unet_crack_best.pt` in `dl_models/`
- **Option B:** Train your own model (see [Training](#training-your-own-model))

### Running Inspection

**1. Configure paths:**
```bash
# Copy template
cp fmvs_inspector/inspection/config_local.py.template fmvs_inspector/inspection/config_local.py

# Edit config_local.py with your paths:
# - video_path: Path to your video
# - log_dir: Where to save detection logs
# - img_dir: Where to save detection images
# - mode: "dl" or "opencv"
```

**2. Run inspection:**
```bash
# Windows
python -m fmvs_inspector

# Linux
python3 -m fmvs_inspector
```

**3. Interactive controls:**
- **During ROI selection:**
  - Left click: Add polygon point
  - Right click: Finish polygon
  - `c`: Clear all points
  - `z`: Undo last point
  - `q`: Cancel

- **During inspection:**
  - `SPACE`: Pause/Resume
  - `q`: Quit
  - `r`: Reselect ROI (when paused)

### Output Files

Outputs are saved to directories specified in your `config_local.py`:

```
logs/
  <video_name>/
    <video_name>_detections.txt      # Detection timestamps

images/
  <video_name>/
    <video_name>_HHMMSSmmm_orig.png  # Original frame
    <video_name>_HHMMSSmmm_insp.png  # Annotated with detections
```

**Log file format:**
```
HH:MM:SS.mmm  (frame=0001234, count=2)
HH:MM:SS.mmm  (frame=0001567, count=1)
```

## Configuration

### Basic Configuration

Edit `fmvs_inspector/inspection/config_local.py`:

```python
from dataclasses import replace
from fmvs_inspector.inspection.config import DEFAULT_CONFIG as BASE

DEFAULT_CONFIG = replace(
    BASE,
    video_path="path/to/video.mp4",
    log_dir="logs",
    img_dir="images",
    mode="dl",  # "dl" or "opencv"
)
```

### Advanced Configuration

```python
DEFAULT_CONFIG = replace(
    BASE,
    # Video & paths
    video_path="video.mp4",
    log_dir="logs",
    img_dir="images",
    
    # Time window (seconds)
    start_sec=0,
    end_sec=None,  # None for full video
    
    # Detection mode
    mode="dl",  # "dl" or "opencv"
    
    # Deep learning settings
    dl=replace(
        BASE.dl,
        model=replace(
            BASE.dl.model,
            ckpt_path="dl_models/unet_crack_best.pt",
            mask_thr=0.45,  # Detection threshold (0-1)
            use_amp=True,   # Mixed precision (faster on GPU)
        ),
        history=2,  # Temporal voting frames
        min_pixels=20,  # Minimum pixels for detection
    ),
    
    # OpenCV settings (fallback mode)
    opencv=replace(
        BASE.opencv,
        percentile=99.6,  # Threshold percentile
        history=3,
        use_clahe=True,
    ),
    
    # Shape filtering (post-processing)
    shape=replace(
        BASE.shape,
        min_area=50,        # Minimum contour area
        min_long_side=40,   # Minimum long side length
        min_aspect=2.0,     # Minimum aspect ratio
    ),
    
    # Debug visualization
    debug=replace(
        BASE.debug,
        show_debug=True,  # Show debug window
    ),
)
```

## Training Your Own Model

See [prepare_dl/README.md](prepare_dl/README.md) for complete training instructions.

**Quick overview:**
1. Collect frames using `collect_dl_dataset.py`
2. Label cracks in CVAT Cloud
3. Prepare dataset splits with `prep_dataset.py`
4. Train U-Net with `train_unet.py`

## Project Structure

```
fmvs_inspector/
├── fmvs_inspector/              # Runtime inspection system
│   ├── config/                  # Configuration types
│   ├── detectors/               # OpenCV + DL detectors
│   │   ├── dl_crack.py          # DL detector wrapper
│   │   ├── dl_unet.py           # U-Net model
│   │   └── opencv_blackhat.py   # OpenCV detector
│   ├── inspection/              # Main inspection loop
│   │   ├── config.py            # Default configuration
│   │   ├── config_local.py.template  # Local config template
│   │   ├── main.py              # Entry point
│   │   └── run_inspection.py    # Core loop
│   ├── io/                      # Logging & image saving
│   ├── roi/                     # ROI selection & tracking
│   ├── utils/                   # Shared utilities
│   └── viz/                     # Debug visualization
├── prepare_dl/                  # Training pipeline (separate)
│   ├── collect_dl_dataset.py    # Frame collection
│   ├── train_unet.py            # Model training
│   ├── prep_dataset.py          # Dataset preparation
│   └── ...
├── dl_models/                   # Trained models (gitignored)
├── requirements.txt             # Core dependencies
└── requirements-torch-cu121.txt # GPU dependencies
```

## Detection Modes

### Deep Learning Mode (Recommended)

Uses U-Net semantic segmentation trained on CVAT-labeled crack masks.

**Advantages:**
- More accurate on complex lighting
- Handles reflections better
- Learns from labeled examples

**Requirements:**
- Trained model (`.pt` file)
- GPU recommended (runs on CPU but slower)

### OpenCV Mode (Fallback)

Uses traditional computer vision (black-hat morphology + percentile thresholding).

**Advantages:**
- No training required
- Works without GPU
- Deterministic

**Use when:**
- No trained model available
- Running on CPU-only machine
- High-contrast cracks on uniform background

## Troubleshooting

### GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch torchvision torchaudio
pip install -r requirements-torch-cu121.txt
```

### Video won't open
- Check video codec (H.264/H.265 recommended)
- Try converting with ffmpeg:
  ```bash
  ffmpeg -i input.mp4 -c:v libx264 output.mp4
  ```

### Out of memory errors
- Reduce `BATCH_SIZE` in training script
- Use smaller `INPUT_SIZE` (e.g., 256 instead of 512)
- Enable `use_amp=True` for mixed precision

### No detections found
- Lower `mask_thr` in DL config (e.g., 0.3 instead of 0.45)
- Check ROI includes crack regions
- Try OpenCV mode for comparison

## Citation

If you use this code in your research, please cite:

```
@software{fmvs_inspector_2026,
  author = {Your Name},
  title = {FMVS Inspector: Deep Learning Crack Detection},
  year = {2026},
  url = {https://github.com/yourusername/fmvs_inspector}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com
