#!/usr/bin/env python3
"""
Setup script for Multi-Task Eye Disease Diagnosis System
This script sets up the environment and verifies installation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\n📦 Creating virtual environment...")
    
    if os.path.exists("venv"):
        print("   Virtual environment already exists")
        return True
    
    success, stdout, stderr = run_command(f"{sys.executable} -m venv venv")
    if success:
        print("✅ Virtual environment created")
        return True
    else:
        print(f"❌ Failed to create virtual environment: {stderr}")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📋 Installing requirements...")
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    success, stdout, stderr = run_command(f"{pip_cmd} install -r requirements.txt")
    if success:
        print("✅ Requirements installed successfully")
        return True
    else:
        print(f"❌ Failed to install requirements: {stderr}")
        return False

def verify_gpu_support():
    """Check GPU availability"""
    print("\n🖥️  Checking GPU support...")
    
    try:
        # Check if we're in the virtual environment
        if os.name == 'nt':
            python_cmd = "venv\\Scripts\\python"
        else:
            python_cmd = "venv/bin/python"
        
        success, stdout, stderr = run_command(
            f'{python_cmd} -c "import torch; print(f\\"CUDA available: {{torch.cuda.is_available()}}\\"); print(f\\"CUDA devices: {{torch.cuda.device_count()}}\\")"'
        )
        
        if success:
            print(f"✅ GPU Check Result:")
            print(f"   {stdout.strip()}")
        else:
            print("⚠️  Could not check GPU support (torch not installed yet)")
            
    except Exception as e:
        print(f"⚠️  Could not check GPU support: {e}")

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "dataset",
        "checkpoints", 
        "runs",
        "logs",
        "evaluation_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directory structure created")

def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    # Check if we can import key modules
    if os.name == 'nt':
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    test_imports = [
        "torch",
        "torchvision", 
        "cv2",
        "numpy",
        "matplotlib",
        "albumentations",
        "tqdm"
    ]
    
    for module in test_imports:
        success, stdout, stderr = run_command(
            f'{python_cmd} -c "import {module}; print(f\\"{module}: OK\\")"'
        )
        
        if success:
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module} - {stderr.strip()}")
            return False
    
    print("✅ All imports successful")
    return True

def create_dataset_structure_info():
    """Create information about dataset structure"""
    dataset_info = """
# IDRiD Dataset Structure Setup

Please organize your IDRiD dataset as follows:

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
│       └── b. Testing Set/
│           └── (same structure as training)
└── B. Disease Grading/
    ├── 1. Original Images/
    │   ├── a. Training Set/          # 413 images (IDRiD_001.jpg to IDRiD_413.jpg)
    │   └── b. Testing Set/           # 103 images (IDRiD_001.jpg to IDRiD_103.jpg)
    └── 2. Groundtruths/
        ├── a. IDRiD_Disease Grading_Training Labels.csv
        └── b. IDRiD_Disease Grading_Testing Labels.csv

Download the IDRiD dataset from:
https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
"""
    
    with open("DATASET_SETUP.md", "w") as f:
        f.write(dataset_info)
    
    print("📋 Created DATASET_SETUP.md with instructions")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📋 Next Steps:")
    print("1. Download and organize the IDRiD dataset (see DATASET_SETUP.md)")
    print("2. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    
    print("\n3. Run training:")
    print("   python train.py")
    print("\n4. Make predictions:")
    print("   python predict.py path/to/image.jpg")
    print("\n5. Evaluate model:")
    print("   python evaluate.py")
    
    print("\n📚 For detailed usage instructions, see README.md")
    print("\n" + "="*70)

def main():
    """Main setup function"""
    print("="*70)
    print("🚀 MULTI-TASK EYE DISEASE DIAGNOSIS SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check GPU support
    verify_gpu_support()
    
    # Create directories
    create_directory_structure()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed!")
        sys.exit(1)
    
    # Create dataset setup info
    create_dataset_structure_info()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()