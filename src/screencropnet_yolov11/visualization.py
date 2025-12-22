"""
Visualization module for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Training metrics visualization
- Confusion matrix plotting
- Detection visualization
- Results dashboard
- Comparison plots
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import seaborn as sns


logger = logging.getLogger(__name__)


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class TrainingVisualizer:
    """Visualize training metrics and progress."""
    
    def __init__(self, output_dir: str):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            history: Dictionary with metrics lists
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        # Loss curves
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # mAP curves
        ax = axes[0, 1]
        if 'mAP50' in history:
            ax.plot(epochs, history['mAP50'], 'g-', label='mAP@50', linewidth=2)
        if 'mAP50_95' in history:
            ax.plot(epochs, history['mAP50_95'], 'b-', label='mAP@50-95', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision & Recall
        ax = axes[1, 0]
        if 'precision' in history:
            ax.plot(epochs, history['precision'], 'b-', label='Precision', linewidth=2)
        if 'recall' in history:
            ax.plot(epochs, history['recall'], 'r-', label='Recall', linewidth=2)
        if 'f1' in history:
            ax.plot(epochs, history['f1'], 'g-', label='F1', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall & F1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        if 'learning_rate' in history:
            ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curves saved to: {save_path}")
        
        return fig
    
    def plot_loss_components(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot individual loss components (box, cls, dfl).
        
        Args:
            history: Dictionary with loss component lists
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history.get('box_loss', [])) + 1)
        
        if 'box_loss' in history:
            ax.plot(epochs, history['box_loss'], 'b-', label='Box Loss', linewidth=2)
        if 'cls_loss' in history:
            ax.plot(epochs, history['cls_loss'], 'r-', label='Class Loss', linewidth=2)
        if 'dfl_loss' in history:
            ax.plot(epochs, history['dfl_loss'], 'g-', label='DFL Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Loss components plot saved to: {save_path}")
        
        return fig


class ConfusionMatrixVisualizer:
    """Visualize confusion matrices."""
    
    @staticmethod
    def plot_confusion_matrix(
        matrix: np.ndarray,
        class_names: List[str],
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot confusion matrix.
        
        Args:
            matrix: Confusion matrix array
            class_names: List of class names
            normalize: Normalize values
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        im = ax.imshow(matrix, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='True Class',
            xlabel='Predicted Class'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, format(matrix[i, j], fmt),
                    ha='center', va='center',
                    color='white' if matrix[i, j] > thresh else 'black',
                    fontsize=8
                )
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")
        
        return fig


class DetectionVisualizer:
    """Visualize detection results."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize detection visualizer.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.colors = self._generate_colors(len(class_names))
    
    def _generate_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors for classes."""
        cmap = plt.cm.get_cmap('tab20')
        return [cmap(i / n)[:3] for i in range(n)]
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        show_confidence: bool = True,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input image (BGR)
            detections: List of detection dictionaries
            show_confidence: Show confidence scores
            line_width: Box line width
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            class_id = det['class_id']
            confidence = det['confidence']
            bbox = det['bbox']
            
            # Get color
            color = tuple(int(c * 255) for c in self.colors[class_id % len(self.colors)])
            color = color[::-1]  # RGB to BGR
            
            # Draw box
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
            
            # Draw label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Label text
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def plot_detection_grid(
        self,
        images: List[np.ndarray],
        results: List[List[Dict]],
        cols: int = 3,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot a grid of detection results.
        
        Args:
            images: List of images
            results: List of detection lists
            cols: Number of columns
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n = len(images)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        for idx, (img, dets) in enumerate(zip(images, results)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Draw detections
            annotated = self.draw_detections(img.copy(), dets)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            ax.imshow(annotated)
            ax.set_title(f"Image {idx+1}: {len(dets)} detections")
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Detection grid saved to: {save_path}")
        
        return fig


class DatasetVisualizer:
    """Visualize dataset statistics."""
    
    @staticmethod
    def plot_class_distribution(
        class_counts: Dict[str, int],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot class distribution bar chart.
        
        Args:
            class_counts: Dictionary of class counts
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha='center', va='bottom',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_bbox_size_distribution(
        widths: List[float],
        heights: List[float],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot bounding box size distribution.
        
        Args:
            widths: List of bbox widths
            heights: List of bbox heights
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Width distribution
        axes[0].hist(widths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Bounding Box Width Distribution')
        
        # Height distribution
        axes[1].hist(heights, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Height')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Bounding Box Height Distribution')
        
        # Aspect ratio distribution
        aspect_ratios = [w / h if h > 0 else 0 for w, h in zip(widths, heights)]
        axes[2].hist(aspect_ratios, bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Aspect Ratio (W/H)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Aspect Ratio Distribution')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Bbox distribution plot saved to: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_image_size_distribution(
        sizes: List[Tuple[int, int]],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot image size distribution.
        
        Args:
            sizes: List of (width, height) tuples
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 2D histogram
        h = ax.hist2d(widths, heights, bins=30, cmap='YlOrRd')
        plt.colorbar(h[3], ax=ax, label='Count')
        
        ax.set_xlabel('Image Width (px)', fontsize=12)
        ax.set_ylabel('Image Height (px)', fontsize=12)
        ax.set_title('Image Size Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Image size distribution saved to: {save_path}")
        
        return fig


class ResultsDashboard:
    """Create comprehensive results dashboard."""
    
    def __init__(self, output_dir: str):
        """
        Initialize results dashboard.
        
        Args:
            output_dir: Directory to save dashboard
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dashboard(
        self,
        training_history: Dict[str, List[float]],
        evaluation_results: Dict[str, Any],
        class_names: List[str],
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create comprehensive results dashboard.
        
        Args:
            training_history: Training metrics history
            evaluation_results: Evaluation metrics
            class_names: List of class names
            save_path: Path to save dashboard
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Training loss
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(training_history.get('train_loss', [])) + 1)
        if 'train_loss' in training_history:
            ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train', linewidth=2)
        if 'val_loss' in training_history:
            ax1.plot(epochs, training_history['val_loss'], 'r-', label='Val', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mAP curves
        ax2 = fig.add_subplot(gs[0, 1])
        if 'mAP50' in training_history:
            ax2.plot(epochs, training_history['mAP50'], 'g-', label='mAP@50', linewidth=2)
        if 'mAP50_95' in training_history:
            ax2.plot(epochs, training_history['mAP50_95'], 'b-', label='mAP@50-95', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Mean Average Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision/Recall
        ax3 = fig.add_subplot(gs[0, 2])
        if 'precision' in training_history:
            ax3.plot(epochs, training_history['precision'], 'b-', label='Precision', linewidth=2)
        if 'recall' in training_history:
            ax3.plot(epochs, training_history['recall'], 'r-', label='Recall', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision & Recall')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Per-class metrics bar chart
        ax4 = fig.add_subplot(gs[1, :2])
        if 'per_class' in evaluation_results:
            per_class = evaluation_results['per_class']
            x = np.arange(len(per_class))
            width = 0.25
            
            ap50 = [c['ap50'] for c in per_class]
            precision = [c.get('precision', 0) for c in per_class]
            recall = [c.get('recall', 0) for c in per_class]
            names = [c['class_name'] for c in per_class]
            
            ax4.bar(x - width, ap50, width, label='AP@50', color='steelblue')
            ax4.bar(x, precision, width, label='Precision', color='forestgreen')
            ax4.bar(x + width, recall, width, label='Recall', color='coral')
            
            ax4.set_xlabel('Class')
            ax4.set_ylabel('Score')
            ax4.set_title('Per-Class Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Overall metrics summary
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        if 'overall' in evaluation_results:
            overall = evaluation_results['overall']
            summary_text = "Final Results\n" + "=" * 25 + "\n\n"
            summary_text += f"mAP@50:      {overall.get('mAP50', 0):.4f}\n"
            summary_text += f"mAP@50-95:   {overall.get('mAP50_95', 0):.4f}\n"
            summary_text += f"Precision:   {overall.get('precision', 0):.4f}\n"
            summary_text += f"Recall:      {overall.get('recall', 0):.4f}\n"
            summary_text += f"F1 Score:    {overall.get('f1', 0):.4f}\n"
            
            ax5.text(
                0.1, 0.5, summary_text,
                transform=ax5.transAxes,
                fontsize=14,
                verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3)
            )
        
        # Confusion matrix
        ax6 = fig.add_subplot(gs[2, :])
        if 'confusion_matrix' in evaluation_results and evaluation_results['confusion_matrix'] is not None:
            matrix = np.array(evaluation_results['confusion_matrix'])
            
            # Normalize
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix_norm = np.divide(matrix, row_sums, where=row_sums != 0)
            
            im = ax6.imshow(matrix_norm, cmap='Blues', aspect='auto')
            plt.colorbar(im, ax=ax6)
            
            ax6.set_xticks(np.arange(len(class_names)))
            ax6.set_yticks(np.arange(len(class_names)))
            ax6.set_xticklabels(class_names, rotation=45, ha='right')
            ax6.set_yticklabels(class_names)
            ax6.set_xlabel('Predicted')
            ax6.set_ylabel('True')
            ax6.set_title('Confusion Matrix (Normalized)')
        
        fig.suptitle('YOLO 11 Twitter Detection - Training Results Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Dashboard saved to: {save_path}")
        
        return fig


def create_comparison_plot(
    results: Dict[str, Dict[str, float]],
    metric: str = 'mAP50_95',
    save_path: Optional[str] = None
) -> Figure:
    """
    Create comparison plot for ablation study results.
    
    Args:
        results: Dictionary mapping config names to metrics
        metric: Metric to compare
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = list(results.keys())
    values = [r.get(metric, 0) for r in results.values()]
    
    bars = ax.bar(configs, values, color=sns.color_palette("viridis", len(configs)))
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Ablation Study: {metric} Comparison', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Highlight best
    best_idx = np.argmax(values)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {save_path}")
    
    return fig
