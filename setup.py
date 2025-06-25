from setuptools import setup, find_packages

setup(
    name="multitask-eye-diagnosis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=8.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
    ],
    author="Research Team",
    description="Modular Multi-Task Vision System for Eye Disease Diagnosis",
    python_requires=">=3.7",
) 