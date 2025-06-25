# Modular Vision-Based Multi-Task Learning for Eye Disease Diagnosis

A comprehensive implementation of a modular deep learning system for multi-task learning in ophthalmology, capable of handling disease classification and lesion segmentation from retinal images using dynamic task-specific routing.

## 🏗️ Architecture Overview

The system implements a unified deep learning framework with the following key components:

- **Shared Feature Extraction Backbone**: ResNet-based feature extractor for common representations
- **Dynamic Routing Mechanism**: Intelligent gating network for task-specific expert selection
- **Modular Expert System**: Specialized processing units for different tasks
- **Multi-Task Learning Framework**: Unified training with combined loss functions

## 📁 Project Structure

```
multitask-eye-diagnosis/
├── main.py                 # Main training and evaluation script
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── models/
│   ├── __init__.py
│   ├── backbone.py         # Shared feature extraction backbone
│   ├── experts.py          # Task-specific expert modules
│   ├── routing.py          # Dynamic routing mechanism
│   └── multitask_model.py  # Complete multi-task model
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset handling and preprocessing
│   └── transforms.py       # Data augmentation and transforms
├── utils/
│   ├── __init__.py
│   ├── losses.py           # Loss functions
│   ├── metrics.py          # Evaluation metrics
│   ├── visualization.py    # Result visualization
│   └── config.py           # Configuration settings
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training pipeline
│   └── evaluator.py        # Evaluation pipeline
├── checkpoints/           # Model checkpoints (created automatically)
├── results/              # Evaluation results (created automatically)
└── data/                 # Dataset directory (created by setup)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── classification_labels/
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── segmentation_masks/
        ├── train/
        ├── val/
        └── test/
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd multitask-eye-diagnosis

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Data Setup

```bash
# Create data structure
python main.py --mode setup
```

This creates the required directory structure. You'll need to:

1. **Add your IDRiD dataset images** to the respective `data/images/train/`, `data/images/val/`, and `data/images/test/` directories
2. **Add segmentation masks** to the `data/segmentation_masks/` directories
3. **Update CSV files** in `data/classification_labels/` with your image names and disease grades

**CSV Format:**
```csv
image_name,grade
IDRiD_01,0
IDRiD_02,2
IDRiD_03,1
...
```

### 3. Training

```bash
# Start training with default configuration
python main.py --mode train

# With custom data path
python main.py --mode train --data_root /path/to/your/data
```

### 4. Evaluation

```bash
# Evaluate the best model
python main.py --mode eval

# Evaluate specific model
python main.py --mode eval --model_path checkpoints/best_model.pth
```

### 5. Inference

```bash
# Run inference on a single image
python main.py --mode inference --input_path path/to/retinal_image.jpg
```

## 🛠️ Configuration

Key configuration parameters in `utils/config.py`:

```python
@dataclass
class Config:
    # Model settings
    backbone: str = "resnet50"          # Backbone architecture
    num_classes: int = 5                # Disease grading classes (0-4)
    num_experts: int = 3                # Number of expert modules
    hidden_dim: int = 256               # Hidden layer dimensions
    dropout_rate: float = 0.2           # Dropout rate
    
    # Training settings
    batch_size: int = 16                # Batch size
    num_epochs: int = 100               # Maximum epochs
    learning_rate: float = 1e-4         # Initial learning rate
    patience: int = 10                  # Early stopping patience
    
    # Loss weights
    classification_weight: float = 1.0   # Classification loss weight
    segmentation_weight: float = 1.0     # Segmentation loss weight
    
    # Data settings
    image_size: Tuple[int, int] = (512, 512)  # Input image size
```

## 📊 Model Architecture Details

### Shared Backbone
- **Architecture**: ResNet-50 (default) or ResNet-34
- **Purpose**: Extract common low-level and mid-level features
- **Pretrained**: ImageNet weights for transfer learning

### Dynamic Router
- **Input**: Shared features + optional task ID
- **Output**: Expert selection weights (softmax probabilities)
- **Architecture**: Global average pooling → FC layers → Softmax

### Expert Modules

1. **Classification Expert**
   - Global average pooling → FC layers → Dropout → Classification head
   - Output: Disease grade probabilities (5 classes)

2. **Segmentation Expert**
   - U-Net style decoder with skip connections
   - Output: Binary lesion segmentation mask

3. **General Expert**
   - Convolutional processing for feature refinement
   - Used for auxiliary tasks or feature enhancement

### Multi-Task Learning
- **Combined Loss**: Weighted sum of classification and segmentation losses
- **Task Balancing**: Dynamic loss weighting to prevent task dominance
- **Optimization**: Adam optimizer with learning rate scheduling

## 📈 Performance Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **AUC**: Area under ROC curve (multi-class OvR)

### Segmentation Metrics
- **Dice Coefficient**: Overlap similarity measure
- **IoU**: Intersection over Union
- **Pixel Accuracy**: Pixel-wise classification accuracy

### Efficiency Metrics
- **Parameter Count**: Total model parameters
- **Inference Time**: Forward pass timing
- **Memory Usage**: GPU memory consumption

## 🔬 Experimental Features

### Dynamic Routing Analysis
The system provides insights into expert utilization:
- Routing weight distributions
- Task-specific expert preferences
- Feature importance analysis

### Ablation Studies
Built-in support for:
- Single-task vs multi-task comparison
- Different backbone architectures
- Various loss function combinations
- Expert module variations

### Visualization Tools
- Training curve plotting
- Prediction overlay visualization
- Confusion matrix generation
- Expert routing analysis

## 📋 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
scikit-learn>=1.0.0
pandas>=1.3.0
```

## 🎯 Key Features

### ✅ Modular Design
- Easy to extend with new tasks
- Pluggable expert modules
- Configurable routing mechanisms

### ✅ Comprehensive Evaluation
- Multiple metrics for each task
- Visual result analysis
- Baseline comparison tools

### ✅ Production Ready
- Checkpoint management
- Reproducible results (fixed seeds)
- Error handling and logging

### ✅ Research Friendly
- Detailed metrics tracking
- Ablation study support
- Visualization tools

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   batch_size: int = 8  # or 4
   ```

2. **Missing Dataset Files**
   ```bash
   # Make sure CSV files have correct format
   # Check image file extensions (.jpg, .png)
   # Verify mask files exist for segmentation
   ```

3. **Poor Performance**
   ```bash
   # Check data quality and labels
   # Adjust loss weights
   # Increase training epochs
   # Verify data augmentation
   ```

### Data Preparation Tips

1. **Image Quality**: Ensure consistent image quality and resolution
2. **Label Accuracy**: Verify disease grading labels are correct
3. **Mask Quality**: Check segmentation masks are properly aligned
4. **Data Balance**: Consider class imbalance in disease grades

## 📚 Research Applications

This implementation supports research in:

- **Multi-Task Learning**: Comparative studies of MTL vs STL
- **Medical Image Analysis**: Ophthalmology-specific applications  
- **Dynamic Routing**: Expert selection mechanisms
- **Transfer Learning**: Pretrained model utilization
- **Model Efficiency**: Parameter sharing strategies

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional expert modules
- New routing mechanisms
- Advanced loss functions
- More evaluation metrics
- Extended visualization tools

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with your institution's policies and relevant data usage agreements when working with medical datasets.

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{multitask-eye-diagnosis,
  title={Modular Vision-Based Multi-Task Learning for Eye Disease Diagnosis},
  author={Research Team},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-repo/multitask-eye-diagnosis}
}
```

## 🆘 Support

For questions and issues:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Research! 🔬✨**