<p align="center">
<h3 align="center">Module C5 Project</h3>

  <p align="center">
    Project for the Module C5-Visual Recognition in Master's in Computer Vision in Barcelona.
<br>
    <a href="https://github.com/carmecorbi/mcv-c5-team1/issues/new?template=bug.md">Report bug</a>
    ·
    <a href="https://github.com/carmecorbi/mcv-c5-team1/issues/new?template=feature.md&labels=feature">Request feature</a>
  </p>
</p>

> [!IMPORTANT]
> The presentation of W2 from this group is available [here](https://docs.google.com/presentation/d/1-VsnziZDube8XFzq-4NXA-_pu9qTRQsRqr77MLSaIro/edit?usp=sharing). If for some reason you don't have permissions to access it, contact any of the administrators of this repository.

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
This is the repository for Group 1 of the `C5. Visual Recognition` module from the Master in Computer Vision (MCV) at Universitat Autonoma de Barcelona (UAB) during the 2025 edition. In here, you can find all the code and experiments performed in order to fulfill the requirements of this project.

For further details, refer to each week's readme files or proceed with the installation of the code material.

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
This repository is organized based on the following structure, which separates each week into a different subfolder:

```bash
mcv-c5-team1/
│── week1/
│   ├── src/  # Source code for week1
│   │   ├── detectron/
│   │   ├── domain_sift/
│   │   ├── hugging_face/
│   │   ├── ultralytics/
│   │   ├── main.py
│   │   └── ...
│   └── README.md # Explanation of week1
│── README.md
│── LICENSE
│── ...
```

<h2>WEEK 1</h2>

The contents of the first week are in the folder `week1`. The `README` file can be found [here](week1/README.md).

<h2>WEEK 2</h2>

The contents of the first week are in the folder `week2`. The `README` file can be found [here](week2/README.md).
## Team Members

This project was developed by the following team members:

- **[Judit Salavedra](https://github.com/juditsalavedra)**
- **[Judith Caldés](https://github.com/judithcaldes)**
- **[Carme Corbi](https://github.com/carmecorbi)**
- **[Yeray Cordero](https://github.com/yeray142)**

## License
The MIT License (MIT). Please see [LICENSE File](LICENSE) for more information.
