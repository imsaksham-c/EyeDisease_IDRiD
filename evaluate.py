# Evaluation script for the multi-task model
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Add src to path
sys.path.append('src')

from config import Config
from dataset import create_data_loaders
from model import MultiTaskModel
from loss import CombinedLoss, MetricsCalculator

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path, config=None):
        self.config = config if config else Config()
        self.device = self.config.device
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Load data
        _, self.test_loader = create_data_loaders(self.config)
        
        # Initialize loss function
        self.criterion = CombinedLoss(self.config).to(self.device)
        
        # Class names
        self.dr_grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        self.dme_risks = ['No DME', 'Mild DME', 'Severe DME']
        self.lesion_types = ['Background', 'Microaneurysms', 'Haemorrhages', 
                           'Hard Exudates', 'Soft Exudates', 'Optic Disc']
    
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = MultiTaskModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        print("Starting evaluation...")
        
        all_predictions = {
            'dr_preds': [], 'dr_targets': [],
            'dme_preds': [], 'dme_targets': [],
            'seg_preds': [], 'seg_targets': [],
            'gates': []
        }
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move data to device
                images = batch['image'].to(self.device)
                dr_grades = batch['dr_grade'].to(self.device)
                dme_risks = batch['dme_risk'].to(self.device)
                seg_masks = batch['seg_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Store predictions
                all_predictions['gates'].extend(predictions['gates'].cpu().numpy())
                
                # Classification predictions
                cls_mask = (dr_grades >= 0) | (dme_risks >= 0)
                if cls_mask.any():
                    dr_valid = dr_grades[dr_grades >= 0]
                    dme_valid = dme_risks[dme_risks >= 0]
                    
                    if len(dr_valid) > 0:
                        dr_preds = predictions['cls_output']['dr_logits'][dr_grades >= 0]
                        all_predictions['dr_preds'].extend(dr_preds.cpu().numpy())
                        all_predictions['dr_targets'].extend(dr_valid.cpu().numpy())
                    
                    if len(dme_valid) > 0:
                        dme_preds = predictions['cls_output']['dme_logits'][dme_risks >= 0]
                        all_predictions['dme_preds'].extend(dme_preds.cpu().numpy())
                        all_predictions['dme_targets'].extend(dme_valid.cpu().numpy())
                
                # Segmentation predictions
                seg_mask = seg_masks.sum(dim=(1, 2, 3)) > 0
                if seg_mask.any():
                    seg_preds = torch.sigmoid(predictions['seg_output'][seg_mask])
                    seg_targets = seg_masks[seg_mask]
                    
                    all_predictions['seg_preds'].extend(seg_preds.cpu().numpy())
                    all_predictions['seg_targets'].extend(seg_targets.cpu().numpy())
                
                # Calculate loss
                targets = {
                    'dr_grade': dr_grades,
                    'dme_risk': dme_risks,
                    'seg_mask': seg_masks
                }
                
                loss, _ = self.criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Average test loss: {avg_loss:.4f}")
        
        # Calculate metrics
        results = self._calculate_metrics(all_predictions)
        results['test_loss'] = avg_loss
        
        return results
    
    def _calculate_metrics(self, predictions):
        """Calculate comprehensive metrics"""
        results = {}
        
        # Classification metrics
        if predictions['dr_preds']:
            dr_preds = np.array(predictions['dr_preds'])
            dr_targets = np.array(predictions['dr_targets'])
            
            dr_pred_classes = np.argmax(dr_preds, axis=1)
            dr_accuracy = (dr_pred_classes == dr_targets).mean()
            
            # Quadratic weighted kappa
            dr_kappa = MetricsCalculator.calculate_quadratic_kappa(
                torch.tensor(dr_preds), torch.tensor(dr_targets), 
                self.config.num_classes_dr
            )
            
            results['dr_classification'] = {
                'accuracy': float(dr_accuracy),
                'quadratic_kappa': float(dr_kappa),
                'confusion_matrix': confusion_matrix(dr_targets, dr_pred_classes).tolist(),
                'classification_report': classification_report(
                    dr_targets, dr_pred_classes, 
                    target_names=self.dr_grades, 
                    output_dict=True
                )
            }
        
        if predictions['dme_preds']:
            dme_preds = np.array(predictions['dme_preds'])
            dme_targets = np.array(predictions['dme_targets'])
            
            dme_pred_classes = np.argmax(dme_preds, axis=1)
            dme_accuracy = (dme_pred_classes == dme_targets).mean()
            
            results['dme_classification'] = {
                'accuracy': float(dme_accuracy),
                'confusion_matrix': confusion_matrix(dme_targets, dme_pred_classes).tolist(),
                'classification_report': classification_report(
                    dme_targets, dme_pred_classes,
                    target_names=self.dme_risks,
                    output_dict=True
                )
            }
        
        # Segmentation metrics
        if predictions['seg_preds']:
            seg_preds = np.array(predictions['seg_preds'])
            seg_targets = np.array(predictions['seg_targets'])
            
            # Calculate metrics for each class
            dice_scores = []
            iou_scores = []
            
            for i in range(seg_preds.shape[1]):
                class_dice = []
                class_iou = []
                
                for j in range(len(seg_preds)):
                    dice = MetricsCalculator.calculate_dice_score(
                        torch.tensor(seg_preds[j, i]),
                        torch.tensor(seg_targets[j, i])
                    )
                    iou = MetricsCalculator.calculate_iou(
                        torch.tensor(seg_preds[j, i]),
                        torch.tensor(seg_targets[j, i])
                    )
                    
                    class_dice.append(dice)
                    class_iou.append(iou)
                
                dice_scores.append(np.mean(class_dice))
                iou_scores.append(np.mean(class_iou))
            
            results['segmentation'] = {
                'mean_dice': float(np.mean(dice_scores)),
                'mean_iou': float(np.mean(iou_scores)),
                'class_dice_scores': {
                    self.lesion_types[i]: float(dice_scores[i]) 
                    for i in range(len(dice_scores))
                },
                'class_iou_scores': {
                    self.lesion_types[i]: float(iou_scores[i]) 
                    for i in range(len(iou_scores))
                }
            }
        
        # Gating statistics
        gates = np.array(predictions['gates'])
        results['gating'] = {
            'avg_classification_gate': float(np.mean(gates[:, 0])),
            'avg_segmentation_gate': float(np.mean(gates[:, 1])),
            'std_classification_gate': float(np.std(gates[:, 0])),
            'std_segmentation_gate': float(np.std(gates[:, 1]))
        }
        
        return results
    
    def create_visualizations(self, results, save_dir):
        """Create evaluation visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # DR confusion matrix
        if 'dr_classification' in results:
            plt.figure(figsize=(8, 6))
            cm = np.array(results['dr_classification']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.dr_grades,
                       yticklabels=self.dr_grades)
            plt.title('DR Classification Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'dr_confusion_matrix.png'), dpi=150)
            plt.close()
        
        # DME confusion matrix
        if 'dme_classification' in results:
            plt.figure(figsize=(6, 5))
            cm = np.array(results['dme_classification']['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.dme_risks,
                       yticklabels=self.dme_risks)
            plt.title('DME Classification Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'dme_confusion_matrix.png'), dpi=150)
            plt.close()
        
        # Segmentation performance
        if 'segmentation' in results:
            seg_results = results['segmentation']
            
            # Dice scores bar plot
            plt.figure(figsize=(12, 6))
            dice_scores = list(seg_results['class_dice_scores'].values())
            lesion_names = list(seg_results['class_dice_scores'].keys())
            
            bars = plt.bar(lesion_names, dice_scores, color='skyblue', alpha=0.7)
            plt.title('Dice Scores by Lesion Type')
            plt.ylabel('Dice Score')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, dice_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'segmentation_dice_scores.png'), dpi=150)
            plt.close()
        
        # Gating distribution
        if 'gating' in results:
            gating_stats = results['gating']
            
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.bar(['Classification', 'Segmentation'], 
                   [gating_stats['avg_classification_gate'], 
                    gating_stats['avg_segmentation_gate']],
                   yerr=[gating_stats['std_classification_gate'],
                         gating_stats['std_segmentation_gate']],
                   color=['lightcoral', 'lightblue'], alpha=0.7)
            plt.title('Average Gating Weights')
            plt.ylabel('Gate Value')
            plt.ylim(0, 1)
            
            plt.subplot(1, 2, 2)
            gates_data = [gating_stats['avg_classification_gate'], 
                         gating_stats['avg_segmentation_gate']]
            plt.pie(gates_data, labels=['Classification', 'Segmentation'], 
                   autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
            plt.title('Task Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'gating_analysis.png'), dpi=150)
            plt.close()
        
        print(f"Visualizations saved to {save_dir}")
    
    def save_results(self, results, save_dir):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON results
        results_path = os.path.join(save_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        # Create visualizations
        self.create_visualizations(results, save_dir)
        
        return results_path
    
    def print_summary(self, results):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Overall loss
        print(f"\nOverall Test Loss: {results['test_loss']:.4f}")
        
        # Classification results
        if 'dr_classification' in results:
            dr_results = results['dr_classification']
            print(f"\n📊 DIABETIC RETINOPATHY CLASSIFICATION:")
            print(f"   Accuracy:         {dr_results['accuracy']:.4f}")
            print(f"   Quadratic Kappa:  {dr_results['quadratic_kappa']:.4f}")
        
        if 'dme_classification' in results:
            dme_results = results['dme_classification']
            print(f"\n📊 DIABETIC MACULAR EDEMA CLASSIFICATION:")
            print(f"   Accuracy:         {dme_results['accuracy']:.4f}")
        
        # Segmentation results
        if 'segmentation' in results:
            seg_results = results['segmentation']
            print(f"\n🎯 SEGMENTATION:")
            print(f"   Mean Dice Score:  {seg_results['mean_dice']:.4f}")
            print(f"   Mean IoU:         {seg_results['mean_iou']:.4f}")
            
            print(f"\n   Per-class Dice Scores:")
            for lesion, score in seg_results['class_dice_scores'].items():
                print(f"     {lesion:15}: {score:.4f}")
        
        # Gating results
        if 'gating' in results:
            gating = results['gating']
            print(f"\n⚡ GATING MECHANISM:")
            print(f"   Avg Classification Gate: {gating['avg_classification_gate']:.3f} ± {gating['std_classification_gate']:.3f}")
            print(f"   Avg Segmentation Gate:   {gating['avg_segmentation_gate']:.3f} ± {gating['std_segmentation_gate']:.3f}")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Multi-task Model Evaluation")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--output_dir", default="evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        return
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(args.model_path)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results and visualizations
        evaluator.save_results(results, args.output_dir)
        
        print(f"\n✅ Evaluation completed! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()