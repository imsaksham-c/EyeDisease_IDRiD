import argparse
import os
import torch
from pathlib import Path

# Import all components
from utils.config import Config, set_seed
from data.dataset import IDRiDDataset
from data.transforms import RetinalTransforms
from models.multitask_model import MultiTaskModel
from training.trainer import Trainer
from training.evaluator import Evaluator


def create_sample_data_structure():
    """Create sample data structure for testing."""
    import pandas as pd

    # Create directories
    data_dirs = [
        "data/images/train",
        "data/images/val",
        "data/images/test",
        "data/segmentation_masks/train",
        "data/segmentation_masks/val",
        "data/segmentation_masks/test",
        "data/classification_labels",
    ]

    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Create sample CSV files
    for split in ["train", "val", "test"]:
        sample_data = {
            "image_name": [f"sample_{i:03d}" for i in range(10)],
            "grade": [i % 5 for i in range(10)],  # Disease grades 0-4
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(f"data/classification_labels/{split}.csv", index=False)

    print("Sample data structure created. Please add your actual IDRiD dataset.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task Vision System for Eye Disease Diagnosis"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "inference", "setup"],
        default="train",
        help="Mode to run",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to trained model"
    )
    parser.add_argument(
        "--input_path", type=str, default=None, help="Path to input image for inference"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/", help="Root directory for data"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()
    config.data_root = args.data_root

    # Set random seed for reproducibility
    set_seed(config.seed)

    if args.mode == "setup":
        create_sample_data_structure()
        return

    # Create model
    model = MultiTaskModel(config).to(config.device)

    if args.mode == "train":
        print("Starting Training Mode")

        # Create datasets
        train_transforms = RetinalTransforms(config.image_size, is_training=True)
        val_transforms = RetinalTransforms(config.image_size, is_training=False)

        try:
            train_dataset = IDRiDDataset(config.data_root, "train", train_transforms)
            val_dataset = IDRiDDataset(config.data_root, "val", val_transforms)

            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")

            # Initialize trainer
            trainer = Trainer(model, train_dataset, val_dataset, config)

            # Start training
            trainer.train()

        except Exception as e:
            print(f"Error in training: {e}")
            print(
                "Make sure your data is properly organized. Run with --mode setup to create sample structure."
            )

    elif args.mode == "eval":
        print("Starting Evaluation Mode")

        if args.model_path is None:
            args.model_path = os.path.join(config.checkpoint_dir, "best_model.pth")

        # Load trained model
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {args.model_path}")
        else:
            print(f"Model file not found: {args.model_path}")
            return

        # Create test dataset
        test_transforms = RetinalTransforms(config.image_size, is_training=False)

        try:
            test_dataset = IDRiDDataset(config.data_root, "test", test_transforms)
            print(f"Test samples: {len(test_dataset)}")

            # Initialize evaluator
            evaluator = Evaluator(model, test_dataset, config)

            # Run evaluation
            results = evaluator.evaluate()

            # Compare with baseline
            evaluator.compare_with_baseline()

        except Exception as e:
            print(f"Error in evaluation: {e}")
            print("Make sure your test data is available.")

    elif args.mode == "inference":
        print("Starting Inference Mode")

        if args.model_path is None:
            args.model_path = os.path.join(config.checkpoint_dir, "best_model.pth")

        if args.input_path is None:
            print("Please provide --input_path for inference")
            return

        # Load trained model
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Model loaded from {args.model_path}")
        else:
            print(f"Model file not found: {args.model_path}")
            return

        # Run inference
        run_inference(model, args.input_path, config)


def run_inference(model, image_path, config):
    """Run inference on a single image."""
    from PIL import Image
    import torch
    import numpy as np

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    # Apply transforms
    transforms = RetinalTransforms(config.image_size, is_training=False)
    image, _ = transforms(image, None)
    image = image.unsqueeze(0).to(config.device)

    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image, task="both")

    # Process results
    if "classification" in outputs:
        cls_pred = torch.softmax(outputs["classification"], dim=1)
        predicted_grade = torch.argmax(cls_pred, dim=1).item()
        confidence = cls_pred[0, predicted_grade].item()

        print(f"Disease Grade Prediction: {predicted_grade}")
        print(f"Confidence: {confidence:.3f}")

    if "segmentation" in outputs:
        seg_pred = outputs["segmentation"].squeeze().cpu().numpy()
        lesion_area = (seg_pred > 0.5).sum() / seg_pred.size

        print(f"Lesion Area Percentage: {lesion_area:.2%}")

    print(f"Inference completed for {image_path}")


if __name__ == "__main__":
    main()
