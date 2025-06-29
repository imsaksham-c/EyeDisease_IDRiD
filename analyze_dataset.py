#!/usr/bin/env python3
"""
Dataset Analysis Script for Multi-Task Eye Disease Diagnosis
Analyzes dataset imbalance and provides recommendations for balanced training.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

from config import Config
from dataset import IDRiDDataset

def analyze_dataset_imbalance(config):
    """Analyze dataset imbalance and provide detailed statistics"""
    
    print("=" * 80)
    print("DATASET IMBALANCE ANALYSIS")
    print("=" * 80)
    
    # Load training dataset
    train_dataset = IDRiDDataset(config, split='train', balance_sampling=False)
    
    # Get dataset statistics
    stats = train_dataset.dataset_stats
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Segmentation samples: {stats['segmentation_samples']}")
    print(f"   Classification samples: {stats['classification_samples']}")
    
    # Task imbalance analysis
    seg_ratio = stats['segmentation_samples'] / stats['total_samples']
    cls_ratio = stats['classification_samples'] / stats['total_samples']
    
    print(f"\n🎯 TASK IMBALANCE:")
    print(f"   Segmentation ratio: {seg_ratio:.3f} ({stats['segmentation_samples']} samples)")
    print(f"   Classification ratio: {cls_ratio:.3f} ({stats['classification_samples']} samples)")
    
    if abs(seg_ratio - cls_ratio) > 0.1:
        print(f"   ⚠️  SIGNIFICANT TASK IMBALANCE DETECTED!")
        print(f"   💡 Recommendation: Enable balanced sampling")
    
    # DR Grade distribution
    if stats['dr_grades']:
        print(f"\n👁️  DR GRADE DISTRIBUTION:")
        dr_grades = dict(stats['dr_grades'])
        total_dr = sum(dr_grades.values())
        
        for grade, count in sorted(dr_grades.items()):
            percentage = (count / total_dr) * 100
            print(f"   Grade {grade}: {count} samples ({percentage:.1f}%)")
        
        # Check for severe imbalance
        max_count = max(dr_grades.values())
        min_count = min(dr_grades.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print(f"   ⚠️  SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}x difference")
            print(f"   💡 Recommendation: Use class weights and focal loss")
    
    # DME Risk distribution
    if stats['dme_risks']:
        print(f"\n💧 DME RISK DISTRIBUTION:")
        dme_risks = dict(stats['dme_risks'])
        total_dme = sum(dme_risks.values())
        
        for risk, count in sorted(dme_risks.items()):
            percentage = (count / total_dme) * 100
            print(f"   Risk {risk}: {count} samples ({percentage:.1f}%)")
        
        # Check for severe imbalance
        max_count = max(dme_risks.values())
        min_count = min(dme_risks.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print(f"   ⚠️  SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}x difference")
    
    # Lesion distribution
    if stats['lesion_counts']:
        print(f"\n🔍 LESION DISTRIBUTION:")
        lesion_names = {
            1: "Microaneurysms",
            2: "Haemorrhages", 
            3: "Hard Exudates",
            4: "Soft Exudates",
            5: "Optic Disc"
        }
        
        lesion_counts = dict(stats['lesion_counts'])
        total_lesion_samples = stats['segmentation_samples']
        
        for lesion_id, count in sorted(lesion_counts.items()):
            lesion_name = lesion_names.get(lesion_id, f"Lesion {lesion_id}")
            percentage = (count / total_lesion_samples) * 100
            print(f"   {lesion_name}: {count} samples ({percentage:.1f}%)")
        
        # Check for rare lesions
        rare_threshold = total_lesion_samples * 0.1  # Less than 10%
        rare_lesions = [lid for lid, count in lesion_counts.items() if count < rare_threshold]
        
        if rare_lesions:
            print(f"   ⚠️  RARE LESIONS DETECTED: {[lesion_names.get(lid, f'Lesion {lid}') for lid in rare_lesions]}")
            print(f"   💡 Recommendation: Use lesion-specific augmentation")
    
    # Sample weight analysis
    if train_dataset.sample_weights:
        weights = train_dataset.sample_weights
        print(f"\n⚖️  SAMPLE WEIGHT ANALYSIS:")
        print(f"   Min weight: {min(weights):.3f}")
        print(f"   Max weight: {max(weights):.3f}")
        print(f"   Mean weight: {np.mean(weights):.3f}")
        print(f"   Std weight: {np.std(weights):.3f}")
        
        # Weight distribution
        weight_ratio = max(weights) / min(weights) if min(weights) > 0 else float('inf')
        print(f"   Weight ratio (max/min): {weight_ratio:.1f}x")
        
        if weight_ratio > 5:
            print(f"   ⚠️  HIGH WEIGHT VARIANCE - some samples will be heavily upsampled")
    
    return stats, train_dataset

def create_visualizations(stats, output_dir="dataset_analysis"):
    """Create visualizations for dataset analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Imbalance Analysis', fontsize=16, fontweight='bold')
    
    # Task distribution
    axes[0, 0].pie([stats['segmentation_samples'], stats['classification_samples']], 
                   labels=['Segmentation', 'Classification'],
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Task Distribution')
    
    # DR Grade distribution
    if stats['dr_grades']:
        dr_grades = dict(stats['dr_grades'])
        axes[0, 1].bar(dr_grades.keys(), dr_grades.values(), color='skyblue')
        axes[0, 1].set_title('DR Grade Distribution')
        axes[0, 1].set_xlabel('DR Grade')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].grid(True, alpha=0.3)
    
    # DME Risk distribution
    if stats['dme_risks']:
        dme_risks = dict(stats['dme_risks'])
        axes[1, 0].bar(dme_risks.keys(), dme_risks.values(), color='lightcoral')
        axes[1, 0].set_title('DME Risk Distribution')
        axes[1, 0].set_xlabel('DME Risk')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Lesion distribution
    if stats['lesion_counts']:
        lesion_names = {
            1: "Microaneurysms",
            2: "Haemorrhages", 
            3: "Hard Exudates",
            4: "Soft Exudates",
            5: "Optic Disc"
        }
        
        lesion_counts = dict(stats['lesion_counts'])
        lesion_labels = [lesion_names.get(lid, f'Lesion {lid}') for lid in lesion_counts.keys()]
        
        axes[1, 1].bar(range(len(lesion_counts)), lesion_counts.values(), color='lightgreen')
        axes[1, 1].set_title('Lesion Distribution')
        axes[1, 1].set_xlabel('Lesion Type')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].set_xticks(range(len(lesion_counts)))
        axes[1, 1].set_xticklabels(lesion_labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_imbalance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Visualizations saved to: {output_dir}/")

def generate_recommendations(stats):
    """Generate specific recommendations for improving training with imbalanced data"""
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMBALANCED DATASET")
    print("=" * 80)
    
    recommendations = []
    
    # Task imbalance recommendations
    seg_ratio = stats['segmentation_samples'] / stats['total_samples']
    if abs(seg_ratio - 0.5) > 0.1:
        recommendations.append({
            'issue': 'Task imbalance between segmentation and classification',
            'solution': 'Enable balanced sampling with WeightedRandomSampler',
            'config': 'config.use_balanced_sampling = True'
        })
    
    # DR Grade imbalance
    if stats['dr_grades']:
        dr_grades = dict(stats['dr_grades'])
        max_dr = max(dr_grades.values())
        min_dr = min(dr_grades.values())
        if max_dr / min_dr > 3:
            recommendations.append({
                'issue': f'DR Grade imbalance ({max_dr/min_dr:.1f}x difference)',
                'solution': 'Use class weights in CrossEntropyLoss',
                'config': 'config.use_class_weights = True'
            })
    
    # DME Risk imbalance
    if stats['dme_risks']:
        dme_risks = dict(stats['dme_risks'])
        max_dme = max(dme_risks.values())
        min_dme = min(dme_risks.values())
        if max_dme / min_dme > 3:
            recommendations.append({
                'issue': f'DME Risk imbalance ({max_dme/min_dme:.1f}x difference)',
                'solution': 'Use class weights in CrossEntropyLoss',
                'config': 'config.use_class_weights = True'
            })
    
    # Rare lesions
    if stats['lesion_counts']:
        total_seg = stats['segmentation_samples']
        rare_threshold = total_seg * 0.1
        rare_lesions = [lid for lid, count in stats['lesion_counts'].items() if count < rare_threshold]
        
        if rare_lesions:
            recommendations.append({
                'issue': f'Rare lesions detected: {rare_lesions}',
                'solution': 'Use advanced augmentation and lesion-specific weighting',
                'config': 'config.use_advanced_augmentation = True'
            })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['issue']}")
        print(f"   💡 {rec['solution']}")
        print(f"   ⚙️  {rec['config']}")
    
    # Training strategy recommendations
    print(f"\n🎯 TRAINING STRATEGY RECOMMENDATIONS:")
    print(f"   1. Use balanced sampling: config.use_balanced_sampling = True")
    print(f"   2. Enable class weights: config.use_class_weights = True")
    print(f"   3. Use advanced augmentation: config.use_advanced_augmentation = True")
    print(f"   4. Consider focal loss for rare classes")
    print(f"   5. Use ReduceLROnPlateau scheduler")
    print(f"   6. Implement early stopping with patience")
    
    # Hyperparameter recommendations
    print(f"\n⚙️  HYPERPARAMETER RECOMMENDATIONS:")
    print(f"   - Learning rate: 1e-4 (reduced for stability)")
    print(f"   - Batch size: 8 (balanced between memory and diversity)")
    print(f"   - Epochs: 100+ (more epochs needed for convergence)")
    print(f"   - Weight decay: 1e-5 (prevent overfitting)")
    
    return recommendations

def main():
    """Main analysis function"""
    
    print("🔍 Starting Dataset Imbalance Analysis...")
    
    # Initialize config
    config = Config()
    
    # Analyze dataset
    stats, dataset = analyze_dataset_imbalance(config)
    
    # Create visualizations
    create_visualizations(stats)
    
    # Generate recommendations
    recommendations = generate_recommendations(stats)
    
    # Save analysis report
    report_path = "dataset_analysis/analysis_report.txt"
    os.makedirs("dataset_analysis", exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("DATASET IMBALANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Segmentation samples: {stats['segmentation_samples']}\n")
        f.write(f"Classification samples: {stats['classification_samples']}\n\n")
        
        if stats['dr_grades']:
            f.write("DR Grade distribution:\n")
            for grade, count in sorted(stats['dr_grades'].items()):
                f.write(f"  Grade {grade}: {count}\n")
            f.write("\n")
        
        if stats['dme_risks']:
            f.write("DME Risk distribution:\n")
            for risk, count in sorted(stats['dme_risks'].items()):
                f.write(f"  Risk {risk}: {count}\n")
            f.write("\n")
        
        if stats['lesion_counts']:
            f.write("Lesion distribution:\n")
            for lesion_id, count in sorted(stats['lesion_counts'].items()):
                f.write(f"  Lesion {lesion_id}: {count}\n")
            f.write("\n")
        
        f.write("Recommendations:\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec['issue']}\n")
            f.write(f"   Solution: {rec['solution']}\n")
            f.write(f"   Config: {rec['config']}\n\n")
    
    print(f"\n📄 Analysis report saved to: {report_path}")
    print("\n✅ Dataset analysis completed!")

if __name__ == "__main__":
    main() 