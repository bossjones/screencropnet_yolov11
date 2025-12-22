"""
Inference module for YOLO 11 Twitter Screenshot Detection.

This module handles:
- Single image inference
- Batch inference
- Video inference
- Result post-processing
- Export to various formats
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import time

import cv2
import numpy as np
from ultralytics import YOLO


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    bbox_normalized: Tuple[float, float, float, float] = None  # Normalized coords
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': {
                'x1': self.bbox[0],
                'y1': self.bbox[1],
                'x2': self.bbox[2],
                'y2': self.bbox[3],
            },
            'bbox_normalized': {
                'x1': self.bbox_normalized[0],
                'y1': self.bbox_normalized[1],
                'x2': self.bbox_normalized[2],
                'y2': self.bbox_normalized[3],
            } if self.bbox_normalized else None,
        }


@dataclass
class InferenceResult:
    """Container for inference results on a single image."""
    image_path: str
    image_size: Tuple[int, int]  # width, height
    detections: List[Detection] = field(default_factory=list)
    inference_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'image_path': self.image_path,
            'image_size': {
                'width': self.image_size[0],
                'height': self.image_size[1],
            },
            'num_detections': len(self.detections),
            'detections': [d.to_dict() for d in self.detections],
            'inference_time_ms': self.inference_time,
        }
    
    def filter_by_confidence(self, min_conf: float) -> 'InferenceResult':
        """Return new result with detections filtered by confidence."""
        filtered = InferenceResult(
            image_path=self.image_path,
            image_size=self.image_size,
            inference_time=self.inference_time
        )
        filtered.detections = [d for d in self.detections if d.confidence >= min_conf]
        return filtered
    
    def filter_by_class(self, class_ids: List[int]) -> 'InferenceResult':
        """Return new result with detections filtered by class."""
        filtered = InferenceResult(
            image_path=self.image_path,
            image_size=self.image_size,
            inference_time=self.inference_time
        )
        filtered.detections = [d for d in self.detections if d.class_id in class_ids]
        return filtered


class InferencePipeline:
    """
    Inference pipeline for YOLO 11 Twitter Screenshot Detection.
    
    Provides methods for running inference on images, batches, and videos.
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to model weights
            class_names: List of class names
            device: Device to run inference on
            conf_threshold: Confidence threshold for filtering
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        
        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Device: {device}, Confidence: {conf_threshold}, IoU: {iou_threshold}")
    
    def predict_image(
        self,
        image: Union[str, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        augment: bool = False
    ) -> InferenceResult:
        """
        Run inference on a single image.
        
        Args:
            image: Image path or numpy array
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold (uses default if None)
            augment: Use test-time augmentation
            
        Returns:
            InferenceResult with detections
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        # Get image path and size
        if isinstance(image, str):
            image_path = image
            img = cv2.imread(image)
            img_size = (img.shape[1], img.shape[0])  # width, height
        else:
            image_path = "numpy_array"
            img_size = (image.shape[1], image.shape[0])
        
        # Run inference
        start_time = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            max_det=self.max_detections,
            augment=augment,
            verbose=False
        )
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Parse results
        result = InferenceResult(
            image_path=image_path,
            image_size=img_size,
            inference_time=inference_time
        )
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bbox coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Calculate normalized coordinates
                norm_bbox = (
                    xyxy[0] / img_size[0],
                    xyxy[1] / img_size[1],
                    xyxy[2] / img_size[0],
                    xyxy[3] / img_size[1],
                )
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf_score,
                    bbox=tuple(xyxy),
                    bbox_normalized=norm_bbox
                )
                result.detections.append(detection)
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        batch_size: int = 16
    ) -> List[InferenceResult]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            conf: Confidence threshold
            iou: IoU threshold
            batch_size: Batch size for processing
            
        Returns:
            List of InferenceResults
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            start_time = time.perf_counter()
            results = self.model.predict(
                source=batch,
                conf=conf,
                iou=iou,
                device=self.device,
                max_det=self.max_detections,
                verbose=False
            )
            batch_time = (time.perf_counter() - start_time) * 1000
            
            # Parse each result
            for j, res in enumerate(results):
                img_path = batch[j] if isinstance(batch[j], str) else f"image_{i+j}"
                
                # Get image size
                if isinstance(batch[j], str):
                    img = cv2.imread(batch[j])
                    img_size = (img.shape[1], img.shape[0])
                else:
                    img_size = (batch[j].shape[1], batch[j].shape[0])
                
                result = InferenceResult(
                    image_path=img_path,
                    image_size=img_size,
                    inference_time=batch_time / len(batch)
                )
                
                if res.boxes is not None:
                    boxes = res.boxes
                    for k in range(len(boxes)):
                        xyxy = boxes.xyxy[k].cpu().numpy()
                        conf_score = float(boxes.conf[k].cpu().numpy())
                        class_id = int(boxes.cls[k].cpu().numpy())
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=conf_score,
                            bbox=tuple(xyxy)
                        )
                        result.detections.append(detection)
                
                all_results.append(result)
            
            logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} images")
        
        return all_results
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        show: bool = False,
        save_frames: bool = False
    ) -> List[InferenceResult]:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video
            conf: Confidence threshold
            iou: IoU threshold
            show: Display video during processing
            save_frames: Save individual frame results
            
        Returns:
            List of InferenceResults per frame
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            result = self.predict_image(frame, conf=conf, iou=iou)
            result.image_path = f"frame_{frame_idx}"
            frame_results.append(result)
            
            # Draw annotations on frame
            annotated = self._draw_detections(frame, result)
            
            # Save frame
            if writer:
                writer.write(annotated)
            
            # Display
            if show:
                cv2.imshow('YOLO 11 Twitter Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed frame {frame_idx}/{total_frames}")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        
        logger.info(f"Video processing complete: {len(frame_results)} frames")
        
        return frame_results
    
    def _draw_detections(
        self,
        image: np.ndarray,
        result: InferenceResult,
        line_thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """Draw detection boxes on image."""
        annotated = image.copy()
        
        # Color palette for classes
        colors = self._get_color_palette()
        
        for det in result.detections:
            # Get color for class
            color = colors[det.class_id % len(colors)]
            
            # Draw box
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1
            )
        
        return annotated
    
    def _get_color_palette(self) -> List[Tuple[int, int, int]]:
        """Get color palette for visualization."""
        return [
            (255, 56, 56),   # Red - profile_info
            (255, 157, 151), # Light red - tweet_text
            (255, 112, 31),  # Orange - engagement_metrics
            (255, 178, 29),  # Yellow - images
            (207, 210, 49),  # Yellow-green - retweet_section
            (72, 249, 10),   # Green - reply_button
            (146, 204, 23),  # Light green - media_container
            (61, 219, 134),  # Teal - hashtags
            (26, 147, 52),   # Dark green - mentions
            (0, 212, 187),   # Cyan - timestamp
            (44, 153, 168),  # Blue-green - verified_badge
            (0, 194, 255),   # Light blue - quote_tweet
        ]


class ResultExporter:
    """Export inference results to various formats."""
    
    @staticmethod
    def to_json(
        results: Union[InferenceResult, List[InferenceResult]],
        output_path: str
    ) -> str:
        """
        Export results to JSON file.
        
        Args:
            results: Single result or list of results
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        if isinstance(results, InferenceResult):
            results = [results]
        
        data = {
            'num_images': len(results),
            'total_detections': sum(len(r.detections) for r in results),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to: {output_path}")
        return output_path
    
    @staticmethod
    def to_coco(
        results: List[InferenceResult],
        output_path: str,
        class_names: List[str]
    ) -> str:
        """
        Export results to COCO format.
        
        Args:
            results: List of inference results
            output_path: Output file path
            class_names: List of class names
            
        Returns:
            Path to saved file
        """
        coco_results = []
        
        for result in results:
            image_id = Path(result.image_path).stem
            
            for det in result.detections:
                x1, y1, x2, y2 = det.bbox
                width = x2 - x1
                height = y2 - y1
                
                coco_results.append({
                    'image_id': image_id,
                    'category_id': det.class_id + 1,  # COCO uses 1-indexed
                    'bbox': [x1, y1, width, height],
                    'score': det.confidence,
                })
        
        with open(output_path, 'w') as f:
            json.dump(coco_results, f, indent=2)
        
        logger.info(f"COCO format results exported to: {output_path}")
        return output_path
    
    @staticmethod
    def to_yolo(
        results: List[InferenceResult],
        output_dir: str
    ) -> str:
        """
        Export results to YOLO format (one .txt per image).
        
        Args:
            results: List of inference results
            output_dir: Output directory
            
        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            # Get image name
            image_name = Path(result.image_path).stem
            txt_path = output_path / f"{image_name}.txt"
            
            with open(txt_path, 'w') as f:
                for det in result.detections:
                    if det.bbox_normalized:
                        x1, y1, x2, y2 = det.bbox_normalized
                    else:
                        # Calculate normalized coordinates
                        x1 = det.bbox[0] / result.image_size[0]
                        y1 = det.bbox[1] / result.image_size[1]
                        x2 = det.bbox[2] / result.image_size[0]
                        y2 = det.bbox[3] / result.image_size[1]
                    
                    # Convert to YOLO format (x_center, y_center, width, height)
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    f.write(f"{det.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        logger.info(f"YOLO format results exported to: {output_dir}")
        return str(output_path)


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.45,
    class_agnostic: bool = False
) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to detections.
    
    Args:
        detections: List of detections
        iou_threshold: IoU threshold for suppression
        class_agnostic: If True, apply NMS across all classes
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_id for d in detections])
    
    # Apply NMS
    if class_agnostic:
        # Apply NMS across all classes
        keep = _nms(boxes, scores, iou_threshold)
    else:
        # Apply NMS per class
        keep = []
        unique_classes = np.unique(classes)
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_indices = np.where(cls_mask)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            
            cls_keep = _nms(cls_boxes, cls_scores, iou_threshold)
            keep.extend(cls_indices[cls_keep].tolist())
    
    return [detections[i] for i in keep]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Standard NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return keep
