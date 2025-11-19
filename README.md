# AI-Powered Image Generator
This project is a Streamlit-based AI Image Generator that allows users to generate images using multiple Stable Diffusion models (SD v1.5, Dreamlike Photoreal, OpenJourney, Waifu Diffusion, etc.).

It is designed to be lightweight, modular, hardware-friendly, and easy to deploy.
## Overview
The application provides:
- A simple UI for text-to-image generation
- Dynamic model selection (no manual app restart)
- Model information via collapsible dropdowns
- Watermark enforcement & safety filtering
- Organized output folders based on date
- Global logging system using CSV
- Research notes, hardware tips & prompt guidelines

## Project Architecture
project/
│── app.py                         # Main Streamlit application
│── /.venv                         # Local Hardware Env
│── /outputs
│    └── YYYYMMDD/                 # Auto-generated folders per date
│    └── logs/            
│        └── generation_logs.csv   # Global logs
│── requirements.txt
│── README.md

## Pipeline Flow
Prompt → Select Model → Load Pipeline → Generate Image → Save Output → Log Entry → Display Results

## Setup & Installations
1. Clone the repository

git clone https://github.com/Prast667/AI-Image-Generator.git
cd AI-Image-Generator

2. Install Miniforge
Download and install Miniforge for your operating system.

3. Create a New Environment & Go Inside the file AI-IMAGE-GENERATOR in MiniForge Terminal
Create a clean environment for your AI project:
conda create -n sd-env python=3.10

Go Inside file project, Examples:
cd C:\Users\prast\Downloads\ai-image-generator

4. Activate the Environment
Before installing anything:
conda activate sd-env

If your terminal shows: (sd-env)
It means you're inside the correct environment.

5. Install PyTorch
For Windows with CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

6. Install Stable Diffusion / Diffusers Requirements

pip install diffusers transformers accelerate safetensors

If a model requires authentication:
huggingface-cli login #Make HuggingFace Account And Enter your Huggingface token in https://huggingface.co/settings/tokens

7. Install Streamlit
pip install streamlit

8. Run Project
streamlit run app.py


## Hardware Requirements
✔ Recommended (Fast GPU)
NVIDIA GPU (RTX 2060, 3060, 4070, 4090 recommended)
6–8 GB VRAM minimum
CUDA 11.8 / 12.x

✔ Minimum (CPU-only)
8 GB RAM
AVX2 processor
Generation time: 20–60+ seconds per image

✔ Best experience
GPU with 12+ GB VRAM
Steps: 25–40
Resolution: 512×512 to 768×768

## Usage Instructions
Run the application:
streamlit run app.py

The UI includes:
- Model selection
- Prompt & Negative Prompt
- Inference steps
- CFG scale
- Seed (optional)
- Watermark toggle
- Safety filter toggle
- Auto-organizing output folders
- Global logging (CSV)

Example prompt: A futuristic neon-lit city with flying cars, ultra-detailed, cinematic perspective, high sharpness
Negative prompt: blurry, bad hands, distorted face, low resolution, watermark

## Technology Stack
_______________________________________________
Component	        Tech
Frontend	        Streamlit
Backend	            PyTorch + Diffusers
Models	            Stable Diffusion variants
Image Processing	PIL
Logging	            pandas (CSV)
Deployment	        Local 
_______________________________________________

## Model Details
- Stable Diffusion v1.5 :
Fast, lightweight, predictable
Good for general-purpose illustration
- Dreamlike Photoreal 2.0
Strong photorealism
Excellent lighting and textures
Best for portraits & realistic scenes
- OpenJourney
Midjourney-inspired style
Dramatic, stylized, fantasy-like
- Waifu Diffusion
Best for anime/manga artwork
Clean lines + vivid colors

## Prompt Engineering Tips
- Be specific about subject, style, lighting, and detail level
- Add camera/lens details for realism
- Use negative prompts to remove unwanted artifacts
- Modify your prompt slightly between generations for better refinement
- For anime: sharper lines + character style tags
- For realism: lighting, aperture, depth of field, and texture details

## Limitations
- CPU mode is significantly slower
- Some models require 6–8 GB VRAM minimum
- May produce artifacts without proper negative prompts
- Model loading times depend on internet + disk speed
- Generating high-res images requires more VRAM

## Future Improvements
- Support for custom model uploads
- Fine-tuning with LoRA / DreamBooth
- Real-time preview or low-res draft mode
- More control over composition (ControlNet)
- Style transfer module
- Async model loading to reduce UI block









