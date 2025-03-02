<p align="center">
<h3 align="center">Module C5 Project</h3>

  <p align="center">
    Project for the Module C5-Visual Recognition in Master's in Computer Vision in Barcelona.
<br>
    <a href="https://github.com/mcv-m6-video/mcv-c6-2025-team1/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://github.com/mcv-m6-video/mcv-c6-2025-team1/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>

Link Final Presentation: 
## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Project Structure](#project-structure)
- [WEEK 1](#week-1)
- [Team Members](#team-members)
- [License](#license)

## Introduction

## Installation

### Prerequisites

- Python >= 3.12
- `pip` or `conda` package managers
- `uv` (optional, but recommended)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/carmecorbi/mcv-c5-team1
   ```

2. **Navigate to the corresponding week's folder:**
   ```bash
   cd week1
   ```
#### Option 1: Using Conda Environment (Recommended)
   ```bash
   # Create and activate conda environment using our provided environment.yml file
   conda env create -f environment.yml
   conda activate mcv-c5-team1

   # Install PyTorch for your CUDA version.
   (uv) pip install torch torchvision torchaudio
   
   # Install Detectron2
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   
   # Install HuggingFace Transformers
   (uv) pip install transformers
   ```

#### Option 2: Using Python Virtual Environment
   ```bash
   # Create and activate virtual environment
   python -m venv env (or uv venv env)
   
   # On Windows:
   .\env\Scripts\activate
   
   # On MacOS/Linux:
   source env/bin/activate
   
   # Install dependencies
   (uv) pip install -r requirements.txt
   
   # Install Detectron2
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   
   # Install HuggingFace Transformers
   (uv) pip install transformers
   ```

> [!NOTE]  
> For detailed Detectron2 installation instructions, please refer to the [official Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
> If you encounter CUDA compatibility issues with Detectron2, you may need to install specific versions compatible with your CUDA version.

Make sure you have either conda or pip package manager installed on your system before proceeding with the installation. 

## Project Structure
<h2>WEEK 1</h2>

The contents of the first week are in the folder `week1`. The `README` file can be found in [here](week1/README.md)

## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.
