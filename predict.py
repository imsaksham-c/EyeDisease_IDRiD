# Prediction script for single image inference
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json

# Add src to path
sys.path.append('src')

from config import Config
from model import MultiTaskModel
from dataset import get_transforms

class Predictor:
    """Predictor class for multi-task inference"""
    
    def __init__(self, model_path, config=None):
        self.config = config if config else Config()
        self.device = self.config.device
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Get transforms
        self.transforms = get_transforms(self.config, split='val')
        
        # Class names
        self.dr_grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        self.dme_risks = ['No DME', 'Mild DME', 'Severe DME']
        self.lesion_types = ['Background', 'Microaneurysms', 'Haemorrhages', 
                           'Hard Exudates', 'Soft Exudates', 'Optic Disc']
        
        # Create runs directory
        self.runs_dir = self.config.runs_dir
        os.makedirs(self.runs_dir, exist_ok=True)
        
    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = MultiTaskModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Resize to model input size
        image = cv2.resize(image, (self.config.image_size, self.config.image_size))
        
        # Apply transforms
        transformed = self.transforms(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def predict(self, image_path):
        """Run inference on single image"""
        # Preprocess image
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        results = self._process_predictions(predictions, original_image)
        results['image_path'] = image_path
        results['timestamp'] = datetime.now().isoformat()
        
        return results
    
    def _process_predictions(self, predictions, original_image):
        """Process model predictions"""
        results = {}
        
        # Process gating values
        gates = predictions['gates'].squeeze(0).cpu().numpy()
        results['gating'] = {
            'classification_gate': float(gates[0]),
            'segmentation_gate': float(gates[1]),
            'dominant_task': 'classification' if gates[0] > gates[1] else 'segmentation'
        }
        
        # Process classification predictions
        cls_output = predictions['cls_output']
        
        # DR grading
        dr_logits = cls_output['dr_logits'].squeeze(0)
        dr_probs = F.softmax(dr_logits, dim=0).cpu().numpy()
        dr_pred = torch.argmax(dr_logits).item()
        
        results['classification'] = {
            'diabetic_retinopathy': {
                'predicted_grade': dr_pred,
                'predicted_class': self.dr_grades[dr_pred],
                'confidence': float(dr_probs[dr_pred]),
                'all_probabilities': {
                    self.dr_grades[i]: float(dr_probs[i]) for i in range(len(self.dr_grades))
                }
            }
        }
        
        # DME risk
        dme_logits = cls_output['dme_logits'].squeeze(0)
        dme_probs = F.softmax(dme_logits, dim=0).cpu().numpy()
        dme_pred = torch.argmax(dme_logits).item()
        
        results['classification']['diabetic_macular_edema'] = {
            'predicted_risk': dme_pred,
            'predicted_class': self.dme_risks[dme_pred],
            'confidence': float(dme_probs[dme_pred]),
            'all_probabilities': {
                self.dme_risks[i]: float(dme_probs[i]) for i in range(len(self.dme_risks))
            }
        }
        
        # Process segmentation predictions
        seg_output = predictions['seg_output'].squeeze(0)  # Remove batch dimension
        seg_probs = torch.sigmoid(seg_output).cpu().numpy()
        
        results['segmentation'] = {}
        
        # Process each lesion type
        for i, lesion_type in enumerate(self.lesion_types):
            mask = seg_probs[i]
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Calculate statistics
            total_pixels = mask.shape[0] * mask.shape[1]
            positive_pixels = np.sum(binary_mask)
            percentage = (positive_pixels / total_pixels) * 100
            
            results['segmentation'][lesion_type] = {
                'presence_score': float(np.max(mask)),
                'average_score': float(np.mean(mask)),
                'percentage_area': float(percentage),
                'pixel_count': int(positive_pixels)
            }
        
        # Store segmentation masks for visualization
        results['_seg_masks'] = seg_probs
        results['_original_image'] = original_image
        
        return results
    
    def save_results(self, results, output_dir=None, save_masks=True, save_visualization=True):
        """Save prediction results"""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(self.runs_dir, f"prediction_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results (excluding numpy arrays)
        json_results = {k: v for k, v in results.items() 
                       if not k.startswith('_')}
        
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {json_path}")
        
        # Save segmentation masks
        if save_masks and '_seg_masks' in results:
            masks_dir = os.path.join(output_dir, 'masks')
            os.makedirs(masks_dir, exist_ok=True)
            
            seg_masks = results['_seg_masks']
            for i, lesion_type in enumerate(self.lesion_types):
                mask = seg_masks[i]
                mask_uint8 = (mask * 255).astype(np.uint8)
                
                # Save as PNG
                mask_path = os.path.join(masks_dir, f'{lesion_type.lower().replace(" ", "_")}.png')
                cv2.imwrite(mask_path, mask_uint8)
                
                # Save binary mask
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                binary_path = os.path.join(masks_dir, f'{lesion_type.lower().replace(" ", "_")}_binary.png')
                cv2.imwrite(binary_path, binary_mask)
            
            print(f"Segmentation masks saved to {masks_dir}")
        
        # Save visualization
        if save_visualization:
            self._create_visualization(results, output_dir)
        
        return output_dir
    
    def _create_visualization(self, results, output_dir):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 12))
        
        original_image = results['_original_image']
        seg_masks = results['_seg_masks']
        
        # Original image
        plt.subplot(2, 4, 1)
        plt.imshow(original_image)
        plt.title('Original Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Classification results text
        plt.subplot(2, 4, 2)
        plt.axis('off')
        
        # Gating information
        gating_info = results['gating']
        classification_info = results['classification']
        
        text_content = f"""GATING RESULTS:
Classification Gate: {gating_info['classification_gate']:.3f}
Segmentation Gate: {gating_info['segmentation_gate']:.3f}
Dominant Task: {gating_info['dominant_task'].title()}

DIABETIC RETINOPATHY:
Grade: {classification_info['diabetic_retinopathy']['predicted_class']}
Confidence: {classification_info['diabetic_retinopathy']['confidence']:.3f}

MACULAR EDEMA:
Risk: {classification_info['diabetic_macular_edema']['predicted_class']}
Confidence: {classification_info['diabetic_macular_edema']['confidence']:.3f}
"""
        
        plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.title('Prediction Results', fontsize=14, fontweight='bold')
        
        # Segmentation masks (excluding background)
        for i in range(1, 6):  # Skip background (index 0)
            plt.subplot(2, 4, i + 2)
            
            mask = seg_masks[i]
            lesion_type = self.lesion_types[i]
            
            # Create overlay
            overlay = original_image.copy()
            overlay = cv2.resize(overlay, (mask.shape[1], mask.shape[0]))
            
            # Apply colormap to mask
            colored_mask = plt.cm.Reds(mask)[:, :, :3]
            
            # Blend with original image
            alpha = 0.6
            overlay = (1 - alpha) * overlay / 255.0 + alpha * colored_mask
            
            plt.imshow(overlay)
            plt.title(f'{lesion_type}\n({results["segmentation"][lesion_type]["percentage_area"]:.1f}%)', 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
        
        # Overall segmentation overlay
        plt.subplot(2, 4, 8)
        
        # Combine all lesion masks (excluding background)
        combined_mask = np.max(seg_masks[1:], axis=0)
        
        overlay = original_image.copy()
        overlay = cv2.resize(overlay, (combined_mask.shape[1], combined_mask.shape[0]))
        
        # Apply colormap
        colored_mask = plt.cm.jet(combined_mask)[:, :, :3]
        
        # Blend
        alpha = 0.5
        overlay = (1 - alpha) * overlay / 255.0 + alpha * colored_mask
        
        plt.imshow(overlay)
        plt.title('All Lesions Combined', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(output_dir, 'visualization.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {viz_path}")
    
    def print_results(self, results):
        """Print results to terminal"""
        print("\n" + "="*80)
        print("MULTI-TASK EYE DISEASE PREDICTION RESULTS")
        print("="*80)
        
        # Image info
        print(f"\nImage: {os.path.basename(results['image_path'])}")
        print(f"Timestamp: {results['timestamp']}")
        
        # Gating results
        print(f"\n📊 GATING MECHANISM:")
        gating = results['gating']
        print(f"   Classification Gate: {gating['classification_gate']:.3f}")
        print(f"   Segmentation Gate:   {gating['segmentation_gate']:.3f}")
        print(f"   Dominant Task:       {gating['dominant_task'].upper()}")
        
        # Classification results
        print(f"\n🔍 CLASSIFICATION RESULTS:")
        cls_results = results['classification']
        
        # DR grading
        dr_info = cls_results['diabetic_retinopathy']
        print(f"   Diabetic Retinopathy:")
        print(f"     Grade:      {dr_info['predicted_grade']} ({dr_info['predicted_class']})")
        print(f"     Confidence: {dr_info['confidence']:.3f}")
        
        # DME risk
        dme_info = cls_results['diabetic_macular_edema']
        print(f"   Diabetic Macular Edema:")
        print(f"     Risk:       {dme_info['predicted_risk']} ({dme_info['predicted_class']})")
        print(f"     Confidence: {dme_info['confidence']:.3f}")
        
        # Segmentation results
        print(f"\n🎯 SEGMENTATION RESULTS:")
        seg_results = results['segmentation']
        
        for lesion_type, info in seg_results.items():
            if lesion_type == 'Background':
                continue
                
            print(f"   {lesion_type}:")
            print(f"     Presence Score: {info['presence_score']:.3f}")
            print(f"     Area Coverage:  {info['percentage_area']:.2f}%")
            print(f"     Pixel Count:    {info['pixel_count']:,}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Multi-task Eye Disease Prediction")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", 
                       help="Path to trained model")
    parser.add_argument("--output_dir", help="Output directory (auto-generated if not provided)")
    parser.add_argument("--no_masks", action="store_true", help="Don't save segmentation masks")
    parser.add_argument("--no_viz", action="store_true", help="Don't save visualization")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    try:
        # Create predictor
        print("Initializing predictor...")
        predictor = Predictor(args.model_path)
        
        # Run prediction
        print(f"Running inference on {args.image_path}...")
        results = predictor.predict(args.image_path)
        
        # Print results to terminal
        predictor.print_results(results)
        
        # Save results
        output_dir = predictor.save_results(
            results, 
            output_dir=args.output_dir,
            save_masks=not args.no_masks,
            save_visualization=not args.no_viz
        )
        
        print(f"\n✅ All outputs saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()