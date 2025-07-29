#!/usr/bin/env python3
"""
Pipeline Analysis and Improvements for NSL Recognition System
Integrating MediaPipe Preprocessing with VideoMAE Training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd

# =====================================
# PART 1: PREPROCESSING IMPROVEMENTS
# =====================================

class EnhancedPreprocessingAnalyzer:
    """Analyze preprocessing results to optimize training"""
    
    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.stats = defaultdict(list)
        
    def analyze_dataset_quality(self):
        """Comprehensive analysis of preprocessed data"""
        
        results = {
            'class_stats': {},
            'detection_analysis': {},
            'quality_metrics': {}
        }
        
        # Load preprocessing summary
        summary_path = self.preprocessed_dir / 'preprocessing_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                
        # Analyze each class
        for class_dir in self.preprocessed_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
                
            class_name = class_dir.name
            class_files = list(class_dir.glob('*.npz'))
            
            detection_rates = []
            enhanced_rates = []
            hand_positions = []
            frame_qualities = []
            
            for npz_file in class_files[:50]:  # Sample analysis
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    
                    if 'metadata' in data:
                        metadata = data['metadata'].item()
                        detection_rates.append(metadata.get('detection_rate', 0))
                        enhanced_rates.append(metadata.get('enhanced_rate', 0))
                    
                    if 'detection_info' in data:
                        det_info = data['detection_info']
                        # Analyze hand positions
                        for det in det_info:
                            if isinstance(det, dict) and det.get('detected'):
                                if 'hand_position' in det:
                                    hand_positions.append(det['hand_position'])
                    
                    # Analyze frame quality
                    frames = data['frames']
                    frame_quality = self._assess_frame_quality(frames)
                    frame_qualities.append(frame_quality)
                    
                except Exception as e:
                    print(f"Error analyzing {npz_file}: {e}")
            
            results['class_stats'][class_name] = {
                'num_samples': len(class_files),
                'avg_detection_rate': np.mean(detection_rates) if detection_rates else 0,
                'avg_enhanced_rate': np.mean(enhanced_rates) if enhanced_rates else 0,
                'detection_std': np.std(detection_rates) if detection_rates else 0,
                'avg_frame_quality': np.mean(frame_qualities) if frame_qualities else 0,
                'hand_position_distribution': dict(pd.Series(hand_positions).value_counts()) if hand_positions else {}
            }
        
        return results
    
    def _assess_frame_quality(self, frames):
        """Assess quality metrics of frames"""
        quality_scores = []
        
        for frame in frames:
            # Check contrast
            contrast = frame.std()
            
            # Check brightness
            brightness = frame.mean()
            
            # Check sharpness (using Laplacian)
            if len(frame.shape) == 3:
                gray = np.mean(frame, axis=2)
            else:
                gray = frame
            laplacian_var = np.var(np.gradient(gray))
            
            # Combined quality score
            quality = (
                min(1.0, contrast / 50) * 0.3 +  # Contrast score
                (1.0 - abs(brightness - 127) / 127) * 0.3 +  # Brightness score
                min(1.0, laplacian_var / 100) * 0.4  # Sharpness score
            )
            quality_scores.append(quality)
        
        return np.mean(quality_scores)
    
    def visualize_preprocessing_analysis(self, save_path='preprocessing_analysis.png'):
        """Create comprehensive visualization of preprocessing results"""
        
        results = self.analyze_dataset_quality()
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Detection rates by class
        ax1 = fig.add_subplot(gs[0, :2])
        classes = list(results['class_stats'].keys())
        detection_rates = [results['class_stats'][c]['avg_detection_rate'] for c in classes]
        enhanced_rates = [results['class_stats'][c]['avg_enhanced_rate'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, detection_rates, width, label='Original Detection', color='#3498db')
        bars2 = ax1.bar(x + width/2, enhanced_rates, width, label='Enhanced Detection', color='#2ecc71')
        
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Detection Rates by Class', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add improvement percentages
        for i, (d, e) in enumerate(zip(detection_rates, enhanced_rates)):
            if d > 0:
                improvement = ((e - d) / d) * 100
                ax1.text(i, max(d, e) + 0.02, f'+{improvement:.0f}%', 
                        ha='center', fontsize=8, color='green' if improvement > 0 else 'red')
        
        # 2. Sample distribution
        ax2 = fig.add_subplot(gs[0, 2])
        sample_counts = [results['class_stats'][c]['num_samples'] for c in classes]
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        ax2.pie(sample_counts, labels=classes, colors=colors, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Sample Distribution', fontsize=14, fontweight='bold')
        
        # 3. Quality metrics heatmap
        ax3 = fig.add_subplot(gs[1, :])
        quality_matrix = []
        metrics = ['Detection Rate', 'Enhanced Rate', 'Frame Quality', 'Detection Std']
        
        for class_name in classes:
            stats = results['class_stats'][class_name]
            quality_matrix.append([
                stats['avg_detection_rate'],
                stats['avg_enhanced_rate'],
                stats['avg_frame_quality'],
                1 - stats['detection_std']  # Inverse for consistency score
            ])
        
        quality_matrix = np.array(quality_matrix).T
        sns.heatmap(quality_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=classes, yticklabels=metrics, ax=ax3,
                   cbar_kws={'label': 'Score'})
        ax3.set_title('Quality Metrics Heatmap', fontsize=14, fontweight='bold')
        
        # 4. Detection improvement distribution
        ax4 = fig.add_subplot(gs[2, 0])
        improvements = [(e - d) * 100 for d, e in zip(detection_rates, enhanced_rates) if d > 0]
        ax4.hist(improvements, bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(improvements), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(improvements):.1f}%')
        ax4.set_xlabel('Improvement (%)')
        ax4.set_ylabel('Count')
        ax4.set_title('Detection Improvement Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Problem classes identification
        ax5 = fig.add_subplot(gs[2, 1])
        problem_threshold = 0.5
        problem_classes = [(c, r) for c, r in zip(classes, enhanced_rates) if r < problem_threshold]
        
        if problem_classes:
            prob_classes, prob_rates = zip(*problem_classes)
            bars = ax5.bar(range(len(prob_classes)), prob_rates, color='#e74c3c')
            ax5.axhline(problem_threshold, color='orange', linestyle='--', 
                       label=f'Threshold: {problem_threshold:.1f}')
            ax5.set_xticks(range(len(prob_classes)))
            ax5.set_xticklabels(prob_classes, rotation=45, ha='right')
            ax5.set_ylabel('Enhanced Detection Rate')
            ax5.set_title('Problem Classes (Low Detection)', fontsize=12, fontweight='bold')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No problem classes identified!', 
                    ha='center', va='center', fontsize=14, color='green')
            ax5.set_title('Problem Classes', fontsize=12, fontweight='bold')
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        summary_text = f"""
        Preprocessing Summary
        {'='*25}
        
        Total Classes: {len(classes)}
        Total Samples: {sum(sample_counts)}
        
        Detection Rates:
          Original: {np.mean(detection_rates):.1%}
          Enhanced: {np.mean(enhanced_rates):.1%}
          Improvement: {np.mean(improvements):.1f}%
        
        Quality Metrics:
          Avg Frame Quality: {np.mean([results['class_stats'][c]['avg_frame_quality'] for c in classes]):.2f}
          Best Class: {classes[np.argmax(enhanced_rates)]} ({max(enhanced_rates):.1%})
          Worst Class: {classes[np.argmin(enhanced_rates)]} ({min(enhanced_rates):.1%})
        
        Recommendations:
          {'✓' if np.mean(enhanced_rates) > 0.7 else '⚠'} Detection quality {'good' if np.mean(enhanced_rates) > 0.7 else 'needs improvement'}
          {'✓' if np.std(sample_counts) / np.mean(sample_counts) < 0.3 else '⚠'} Class balance {'good' if np.std(sample_counts) / np.mean(sample_counts) < 0.3 else 'needs attention'}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='lightgray', alpha=0.3))
        
        plt.suptitle('Preprocessing Quality Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return results


# =====================================
# PART 2: TRAINING IMPROVEMENTS
# =====================================

class AdaptiveTrainingStrategy:
    """Adaptive training based on preprocessing quality"""
    
    def __init__(self, preprocessing_results):
        self.preprocessing_results = preprocessing_results
        
    def get_class_weights(self):
        """Calculate class weights based on detection quality and sample count"""
        
        class_stats = self.preprocessing_results['class_stats']
        
        weights = {}
        for class_name, stats in class_stats.items():
            # Factor 1: Sample count (inverse frequency)
            sample_weight = 1.0 / max(stats['num_samples'], 1)
            
            # Factor 2: Detection quality (lower quality = higher weight)
            quality_weight = 2.0 - stats['avg_enhanced_rate']
            
            # Factor 3: Consistency (higher std = higher weight)
            consistency_weight = 1.0 + stats['detection_std']
            
            # Combined weight
            weights[class_name] = sample_weight * quality_weight * consistency_weight
        
        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v/max_weight for k, v in weights.items()}
        
        return weights
    
    def get_augmentation_strategy(self, class_name):
        """Get class-specific augmentation strategy"""
        
        stats = self.preprocessing_results['class_stats'].get(class_name, {})
        detection_rate = stats.get('avg_enhanced_rate', 0.5)
        
        # More aggressive augmentation for classes with poor detection
        if detection_rate < 0.4:
            return {
                'mixup_alpha': 0.2,
                'cutmix_prob': 0.3,
                'temporal_shift_prob': 0.4,
                'color_jitter_intensity': 0.2,
                'brightness_range': (0.7, 1.3),
                'use_pose_augmentation': True
            }
        elif detection_rate < 0.7:
            return {
                'mixup_alpha': 0.1,
                'cutmix_prob': 0.1,
                'temporal_shift_prob': 0.3,
                'color_jitter_intensity': 0.15,
                'brightness_range': (0.85, 1.15),
                'use_pose_augmentation': False
            }
        else:
            # Light augmentation for good quality data
            return {
                'mixup_alpha': 0.05,
                'cutmix_prob': 0.0,
                'temporal_shift_prob': 0.2,
                'color_jitter_intensity': 0.1,
                'brightness_range': (0.9, 1.1),
                'use_pose_augmentation': False
            }
    
    def get_learning_rate_schedule(self, base_lr=2e-5):
        """Adaptive learning rate based on data quality"""
        
        avg_detection = np.mean([
            stats['avg_enhanced_rate'] 
            for stats in self.preprocessing_results['class_stats'].values()
        ])
        
        if avg_detection < 0.5:
            # Poor detection - use lower LR and longer warmup
            return {
                'initial_lr': base_lr * 0.5,
                'warmup_ratio': 0.2,
                'schedule': 'cosine_with_restarts',
                'num_cycles': 3
            }
        elif avg_detection < 0.7:
            # Medium detection
            return {
                'initial_lr': base_lr * 0.75,
                'warmup_ratio': 0.15,
                'schedule': 'cosine',
                'num_cycles': 1
            }
        else:
            # Good detection
            return {
                'initial_lr': base_lr,
                'warmup_ratio': 0.1,
                'schedule': 'cosine',
                'num_cycles': 1
            }


# =====================================
# PART 3: INTEGRATED PIPELINE OPTIMIZER
# =====================================

class PipelineOptimizer:
    """Optimize the complete pipeline based on analysis"""
    
    def __init__(self, preprocessed_dir, model_dir):
        self.preprocessed_dir = Path(preprocessed_dir)
        self.model_dir = Path(model_dir)
        
        # Analyze preprocessing
        self.prep_analyzer = EnhancedPreprocessingAnalyzer(preprocessed_dir)
        self.prep_results = self.prep_analyzer.analyze_dataset_quality()
        
        # Setup adaptive training
        self.training_strategy = AdaptiveTrainingStrategy(self.prep_results)
        
    def generate_optimized_config(self):
        """Generate optimized configuration for training"""
        
        # Base configuration
        config = {
            'model_name': 'MCG-NJU/videomae-base',
            'num_classes': len(self.prep_results['class_stats']),
            'batch_size': 2,  # Keep small for M3 Pro
            'epochs': 50,
            'num_folds': 3,
            'seed': 42,
            'save_dir': str(self.model_dir),
            'num_workers': 0,  # macOS compatibility
            'use_wandb': False
        }
        
        # Get adaptive parameters
        lr_schedule = self.training_strategy.get_learning_rate_schedule()
        config.update(lr_schedule)
        
        # Calculate class weights
        class_weights = self.training_strategy.get_class_weights()
        config['class_weights'] = class_weights
        
        # Determine freezing strategy based on data quality
        avg_detection = np.mean([
            stats['avg_enhanced_rate'] 
            for stats in self.prep_results['class_stats'].values()
        ])
        
        if avg_detection < 0.6:
            config['freeze_backbone_layers'] = 10  # Freeze more for poor data
        elif avg_detection < 0.8:
            config['freeze_backbone_layers'] = 8
        else:
            config['freeze_backbone_layers'] = 6
        
        # Early stopping patience
        config['patience'] = 15 if avg_detection < 0.6 else 10
        
        # Gradient clipping
        config['grad_clip'] = 1.0 if avg_detection > 0.7 else 0.5
        
        return config
    
    def identify_preprocessing_issues(self):
        """Identify specific issues in preprocessing"""
        
        issues = []
        recommendations = []
        
        for class_name, stats in self.prep_results['class_stats'].items():
            # Check detection rate
            if stats['avg_enhanced_rate'] < 0.3:
                issues.append(f"{class_name}: Very low detection rate ({stats['avg_enhanced_rate']:.1%})")
                recommendations.append(f"Consider re-recording {class_name} videos with better hand visibility")
            
            elif stats['avg_enhanced_rate'] < 0.5:
                issues.append(f"{class_name}: Low detection rate ({stats['avg_enhanced_rate']:.1%})")
                recommendations.append(f"Review {class_name} videos for lighting/positioning issues")
            
            # Check consistency
            if stats['detection_std'] > 0.3:
                issues.append(f"{class_name}: Inconsistent detection (std: {stats['detection_std']:.2f})")
                recommendations.append(f"Check for varying quality in {class_name} videos")
            
            # Check frame quality
            if stats['avg_frame_quality'] < 0.5:
                issues.append(f"{class_name}: Poor frame quality ({stats['avg_frame_quality']:.2f})")
                recommendations.append(f"Improve lighting/camera settings for {class_name}")
        
        return issues, recommendations
    
    def suggest_improvements(self):
        """Generate improvement suggestions"""
        
        suggestions = []
        
        # Analyze overall statistics
        all_detection_rates = [
            stats['avg_enhanced_rate'] 
            for stats in self.prep_results['class_stats'].values()
        ]
        
        avg_detection = np.mean(all_detection_rates)
        std_detection = np.std(all_detection_rates)
        
        # Detection quality suggestions
        if avg_detection < 0.5:
            suggestions.append("CRITICAL: Overall detection quality is poor. Consider:")
            suggestions.append("  1. Adjusting camera angle to better capture signing space")
            suggestions.append("  2. Improving lighting conditions")
            suggestions.append("  3. Using a plain background")
            suggestions.append("  4. Ensuring hands are clearly visible throughout gestures")
        
        elif avg_detection < 0.7:
            suggestions.append("WARNING: Detection quality needs improvement:")
            suggestions.append("  1. Review videos with detection rate < 50%")
            suggestions.append("  2. Consider re-recording problematic classes")
            suggestions.append("  3. Adjust signing_space_bounds in preprocessing")
        
        # Consistency suggestions
        if std_detection > 0.2:
            suggestions.append("\nINCONSISTENCY DETECTED:")
            suggestions.append("  1. Standardize recording conditions across classes")
            suggestions.append("  2. Use consistent camera positioning")
            suggestions.append("  3. Maintain uniform distance from camera")
        
        # Class balance suggestions
        sample_counts = [
            stats['num_samples'] 
            for stats in self.prep_results['class_stats'].values()
        ]
        
        if max(sample_counts) / min(sample_counts) > 2:
            suggestions.append("\nCLASS IMBALANCE DETECTED:")
            suggestions.append("  1. Augment underrepresented classes")
            suggestions.append("  2. Use weighted sampling in training")
            suggestions.append("  3. Consider collecting more data for small classes")
        
        # Training suggestions based on quality
        if avg_detection > 0.8:
            suggestions.append("\nTRAINING RECOMMENDATIONS (High Quality Data):")
            suggestions.append("  1. Can use higher learning rate (3e-5)")
            suggestions.append("  2. Reduce augmentation intensity")
            suggestions.append("  3. Train for fewer epochs with early stopping")
        else:
            suggestions.append("\nTRAINING RECOMMENDATIONS (Lower Quality Data):")
            suggestions.append("  1. Use lower learning rate (1e-5)")
            suggestions.append("  2. Increase augmentation and regularization")
            suggestions.append("  3. Train for more epochs with patience")
            suggestions.append("  4. Consider ensemble methods")
        
        return suggestions


# =====================================
# PART 4: USAGE EXAMPLE
# =====================================

def analyze_and_optimize_pipeline(preprocessed_dir='mediapipe_preprocessed', 
                                 model_dir='models/videomae_nsl_optimized'):
    """Complete pipeline analysis and optimization"""
    
    print("="*60)
    print("PIPELINE ANALYSIS AND OPTIMIZATION")
    print("="*60)
    
    # Initialize optimizer
    optimizer = PipelineOptimizer(preprocessed_dir, model_dir)
    
    # Visualize preprocessing analysis
    print("\n1. Analyzing preprocessing quality...")
    optimizer.prep_analyzer.visualize_preprocessing_analysis()
    
    # Identify issues
    print("\n2. Identifying issues...")
    issues, recommendations = optimizer.identify_preprocessing_issues()
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("  ✓ No major issues found!")
    
    # Generate suggestions
    print("\n3. Generating improvement suggestions...")
    suggestions = optimizer.suggest_improvements()
    for suggestion in suggestions:
        print(suggestion)
    
    # Generate optimized config
    print("\n4. Generating optimized training configuration...")
    optimized_config = optimizer.generate_optimized_config()
    
    print("\nOPTIMIZED CONFIGURATION:")
    print(json.dumps(optimized_config, indent=2))
    
    # Save configuration
    config_path = Path(model_dir) / 'optimized_config.json'
    config_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_path}")
    
    return optimizer, optimized_config


if __name__ == "__main__":
    # Run complete analysis
    optimizer, config = analyze_and_optimize_pipeline()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the preprocessing analysis visualization")
    print("2. Address any identified issues")
    print("3. Use the optimized configuration for training")
    print("4. Monitor training metrics closely for classes with low detection rates")