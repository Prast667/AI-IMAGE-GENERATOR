# AI-Powered Image Generator

## Overview
Project demo: text-to-image generator using open-source Stable Diffusion (runwayml/stable-diffusion-v1-5).
Built with Python, PyTorch, Diffusers, and Streamlit.

## Features
- Generate images from text prompts
- Simple and Advanced modes
- Style guidance, negative prompt, CFG (guidance), steps, seed
- Progress bar + ETA
- Save images with metadata (JSON)
- Export PNG or JPEG
- Watermarking and basic content filtering
- Local GPU (CUDA) support and CPU fallback

## Requirements
- GPU: NVIDIA with CUDA (tested RTX 4050, CUDA 12.x)
- Python 3.10 recommended
- Miniforge/conda recommended for Windows

## Install (conda recommended)
```bash
# create env
conda create -n sd-env python=3.10 -y
conda activate sd-env

# install pyarrow if needed
conda install -c conda-forge pyarrow -y

# install pytorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install other deps
pip install -r requirements.txt
