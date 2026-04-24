# Deep Learning Lab - NIT Surat
# Hitarth Shah - U23AI048
A comprehensive repository containing deep learning laboratory experiments, projects, and implementations covering various neural network architectures and techniques (from foundational MLPs to advanced Generative Models and Federated Learning).

## 📁 Repository Structure

```
DL-Lab/
├── Lab1/                  # Foundational deep learning concepts (MLP on MNIST)
├── Lab2/                  # Neural networks and training techniques (CNN vs MLP)
├── Lab3/                  # CNN architectures, Focal Loss, and t-SNE embeddings
├── Lab4/                  # Advanced Image Classification (Vision models using timm)
├── Lab5/                  # Model Evaluation & Metrics Analysis
├── Lab6/                  # Generative Models (GANs/Autoencoders) and Image Quality Metrics
├── DL-MinorProject.ipynb  # Classification of Imbalanced WCE Datasets
├── DL-MajorProject/       # BLIP FedMix Medical Image Captioning
└── README.md              # This file
```

## 🧪 Lab Descriptions

### Lab 1: Foundational Concepts
Introduction to deep learning fundamentals. Implements basic Multi-Layer Perceptron (MLP) networks on the MNIST dataset using TensorFlow.

### Lab 2: Convolutional Neural Networks (CNN) Basics
Exploration of Convolutional Neural Networks. Compares the architecture, performance, and training of CNNs versus traditional MLPs on image data (MNIST) using TensorFlow.

### Lab 3: Advanced CNN Architectures
**Multi-part series covering CNN architectures using PyTorch:**
- **Part 1**: CNN architecture comparisons on CIFAR-10.
- **Part 2**: Implementation of VGGNet architectures and advanced loss functions like Focal Loss.
- **Part 3**: Feature extraction and visualizing high-dimensional embeddings using t-SNE.

### Lab 4: Advanced Image Classification
Deep dive into advanced computer vision models using the `timm` (PyTorch Image Models) library to implement and train state-of-the-art vision architectures.

### Lab 5: Model Evaluation & Metrics
Focuses on evaluating deep learning models. Implements advanced classification reports, confusion matrices, and uses t-SNE for plotting dataset and feature distributions using PyTorch.

### Lab 6: Generative Models
Exploration of Generative Deep Learning. Implements GANs and Autoencoders on FashionMNIST/EMNIST datasets. Includes evaluation using specialized image generation metrics such as Frechet Inception Distance (FID), Inception Score (IS), and Structural Similarity Index (SSIM).

## 📊 Projects

### Minor Project: Classification of Imbalanced WCE Datasets
**Located in `DL-MinorProject.ipynb`**
A focused implementation dealing with real-world data challenges in Wireless Capsule Endoscopy (WCE) datasets. 
- **Techniques Used**: Transfer Learning, extensive Data Augmentation, Under/Over-sampling strategies, and Intelligent Learning Rate scheduling to handle severe class imbalances.

### Major Project: Federated BLIP Models
**Located in `DL-MajorProject/`**
An advanced federated learning pipeline for Medical Image Captioning.
- **BLIP Base**: Standard implementation of the BLIP Vision-Language model.
- **BLIP FedMix**: A novel federated learning variant of BLIP optimized for decentralized medical data.
- **Results**: Execution logs, checkpoints, and evaluation metrics (CIDEr, BLEU) are tracked across federated rounds.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- **Key Libraries**: TensorFlow, PyTorch, torchvision, timm, torchinfo, NumPy, Pandas, Matplotlib, scikit-learn.

### Installation
1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install jupyter tensorflow torch torchvision torchaudio timm scikit-learn matplotlib pandas
   ```
3. Open any notebook:
   ```bash
   jupyter notebook
   ```

## 📝 Usage
Each lab and project is contained in a Jupyter Notebook (`.ipynb` file). Simply navigate to the desired directory, open the notebook, and execute the cells sequentially to reproduce the experiments.

## 👨‍💼 Author
Deep Learning Lab - NIT Surat
