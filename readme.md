# Modular Vision-Based Multi-task Learning for Eye Disease Diagnosis

A comprehensive deep learning system for simultaneous eye disease classification and lesion segmentation using the IDRiD dataset.

---

## How to Run

**Quickstart:**

```bash
python eye_disease_multitask.py
```

This will:
- Reorganize the dataset automatically (if needed)
- Train the multi-task model
- Train single-task baseline models
- Evaluate all models
- Generate performance comparison
- Create visualizations and reports

---

## Overview

This project implements a modular multi-task learning architecture that efficiently handles both disease grading classification and lesion segmentation from retinal images. The system uses dynamic task-specific routing within a unified vision model to simulate intelligent task-aware learning in medical imaging.

## Features

- **Modular Architecture**: ResNet-50 backbone with specialized expert modules
- **Dynamic Routing**: Gating networks for intelligent expert selection
- **Multi-task Learning**: Simultaneous classification and segmentation
- **Comprehensive Evaluation**: Performance comparison with single-task baselines
- **Visualization Tools**: Results visualization and training monitoring
- **Reproducible**: Fixed random seeds and organized code structure

## Requirements

### Dependencies

Install the following Python packages (Python 3.7+ recommended):

```bash
pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.21.0 pandas>=1.3.0 opencv-python>=4.5.0 Pillow>=8.3.0 matplotlib>=3.4.0 scikit-learn>=1.0.0 tqdm>=4.62.0
```

Or create a `requirements.txt` with the above and run:
```bash
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: At least 5GB free space for dataset and models

## Dataset Structure

The IDRiD dataset should be organized as follows:

```
dataset/
├── A. Segmentation/
│   ├── 1. Original Images/
│   │   ├── a. Training Set/          # 54 images (IDRiD_01.jpg to IDRiD_54.jpg)
│   │   └── b. Testing Set/           # 27 images (IDRiD_55.jpg to IDRiD_81.jpg)
│   └── 2. All Segmentation Groundtruths/
│       ├── a. Training Set/
│       │   ├── 1. Microaneurysms/    # *_MA.tif files
│       │   ├── 2. Haemorrhages/      # *_HE.tif files
│       │   ├── 3. Hard Exudates/     # *_EX.tif files
│       │   ├── 4. Soft Exudates/     # *_SE.tif files
│       │   └── 5. Optic Disc/        # *_OD.tif files
│       └── b. Testing Set/           # Same structure as training
└── B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/          # 413 images (IDRiD_001.jpg to IDRiD_413.jpg)
    │   └── b. Testing Set/           # 103 images (IDRiD_001.jpg to IDRiD_103.jpg)
    └── 2. Groundtruths/
        ├── a. IDRiD_Disease Grading_Training Labels.csv
        └── b. IDRiD_Disease Grading_Testing Labels.csv
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EyeDisease_IDRiD
```

2. Install dependencies as above.

3. Download and extract the IDRiD dataset to the `dataset/` directory.

## Usage

### Basic Training and Evaluation

```bash
python eye_disease_multitask.py
```

### Configuration

Modify the configuration parameters in the `main()` function in `eye_disease_multitask.py`:

```python
# Configuration parameters
DATASET_PATH = "dataset"          # Path to IDRiD dataset
BATCH_SIZE = 16                   # Batch size for training
NUM_EPOCHS = 50                   # Number of training epochs
LEARNING_RATE = 1e-4              # Learning rate
DEVICE = torch.device('cuda')     # Device for training
```

### Custom Training (as a module)

```python
from eye_disease_multitask import ModularMultiTaskModel, Trainer

# Initialize model
model = ModularMultiTaskModel(
    num_classes_classification=5,
    num_experts=3,
    hidden_dim=512
)

# Create trainer
trainer = Trainer(model, train_loader, val_loader, device, lr=1e-4)

# Train model
trainer.train(num_epochs=50)
```

## Model Architecture

### Multi-task Model Components

1. **Shared Backbone**: ResNet-50 pretrained on ImageNet
2. **Expert Modules**: 3 specialized processing units for each task
3. **Gating Networks**: Dynamic routing mechanism for expert selection
4. **Task Selector**: Learns to balance between classification and segmentation

### Expert Architecture

- **Classification Experts**: Fully connected layers with dropout
- **Segmentation Experts**: Transpose convolutional layers for upsampling
- **Gating Mechanism**: Softmax-based attention for expert selection

## Output Files

The system generates several output files:

- `best_model.pth`: Best performing model weights
- `final_results.json`: Complete evaluation results
- `training_history.json`: Training loss history
- `results_visualization.png`: Sample predictions visualization
- `training_history.png`: Training progress plots
- `confusion_matrix.png`: Classification confusion matrix
- `expert_analysis.png`: Expert utilization analysis

## Performance Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed classification performance
- **Per-class Precision/Recall**: Class-specific performance

### Segmentation Metrics
- **IoU (Intersection over Union)**: Segmentation overlap metric
- **Dice Score**: Segmentation similarity metric
- **Pixel Accuracy**: Per-pixel classification accuracy

## Evaluation Results

The system provides comprehensive performance analysis:

1. **Multi-task vs Single-task Comparison**
2. **Task-specific Performance Metrics**
3. **Expert Utilization Analysis**
4. **Training Convergence Analysis**
5. **Inference Time Profiling**

## Key Features

### Data Preprocessing
- Automatic dataset reorganization
- Image resizing and normalization
- Data augmentation for training
- Balanced sampling for multi-task learning

### Training Pipeline
- Combined loss function with task weighting
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping for stability

### Evaluation Framework
- Comprehensive metrics calculation
- Visual result generation
- Performance comparison tools
- Statistical significance testing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Dataset Loading Errors**
   - Verify dataset structure
   - Check file permissions
   - Ensure all required files are present

3. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Check data preprocessing

### Performance Optimization

1. **Memory Optimization**
   - Use DataLoader with multiple workers
   - Enable pin_memory for faster GPU transfer
   - Implement gradient accumulation for large batches

2. **Training Speed**
   - Use mixed precision training
   - Enable CUDA optimizations
   - Optimize data loading pipeline

## Model Interpretation

The system provides several interpretation tools:

1. **Expert Utilization**: Visualization of which experts are used for different inputs
2. **Task Routing**: Analysis of task-specific routing decisions
3. **Feature Visualization**: Activation maps and attention visualizations
4. **Performance Analysis**: Detailed breakdown of model performance

## Extending the System

### Adding New Tasks
1. Create new expert modules
2. Implement task-specific loss functions
3. Update the gating mechanism
4. Modify the evaluation pipeline

### Modifying Architecture
1. Change backbone network
2. Adjust expert module design
3. Implement new routing strategies
4. Add regularization techniques

## Citation

If you use this code in your research, please cite:

```bibtex
@article{multitask_eye_diagnosis,
  title={Modular Vision-Based Multi-task Learning for Eye Disease Diagnosis},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- IDRiD dataset creators: Prasanna Porwal, Samiksha Pachade, and Manesh Kokare
- PyTorch team for the deep learning framework
- torchvision team for pretrained models

## Contact

For questions or issues, please open an issue on the GitHub repository or contact [your-email@example.com].

---

**Note**: This implementation is for research purposes. For clinical applications, please ensure proper validation and regulatory approval.
--- 