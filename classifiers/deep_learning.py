import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image
import logging
from typing import List, Dict, Any, Tuple
import time

from models.pretrained_weights import ModelWeights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchitecturalCNN(nn.Module):
    """CNN model for architectural element classification"""
    
    def __init__(self, num_classes=7):
        super(ArchitecturalCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Assuming 224x224 input -> 14x14 after pooling
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class VectorToImageConverter:
    """Converts vector data to images for CNN processing"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.padding = 20  # Padding around elements
    
    def convert_element_to_image(self, element: Dict[str, Any], 
                               context_elements: List[Dict[str, Any]] = None) -> np.ndarray:
        """Convert a single element to an image representation"""
        
        # Create blank image
        img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 255
        
        # Calculate bounds for proper scaling
        bounds = self._calculate_element_bounds(element)
        if context_elements:
            context_bounds = [self._calculate_element_bounds(elem) for elem in context_elements]
            all_bounds = [bounds] + context_bounds
            combined_bounds = self._combine_bounds(all_bounds)
        else:
            combined_bounds = bounds
        
        # Scale and translate to fit image
        scale_factor = self._calculate_scale_factor(combined_bounds)
        offset = self._calculate_offset(combined_bounds, scale_factor)
        
        # Draw context elements in light gray
        if context_elements:
            for ctx_elem in context_elements:
                self._draw_element(img, ctx_elem, scale_factor, offset, color=(200, 200, 200), thickness=1)
        
        # Draw main element in black
        self._draw_element(img, element, scale_factor, offset, color=(0, 0, 0), thickness=2)
        
        return img
    
    def _calculate_element_bounds(self, element: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Calculate bounding box of an element"""
        element_type = element.get('type', '')
        
        if element_type == 'rectangle':
            center = element.get('center', [0, 0])
            width = element.get('width', 1)
            height = element.get('height', 1)
            
            return (
                center[0] - width/2,
                center[1] - height/2,
                center[0] + width/2,
                center[1] + height/2
            )
        
        elif element_type in ['polyline', 'polygon']:
            points = element.get('points', [[0, 0]])
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        elif element_type == 'line':
            geometry = element.get('geometry', {})
            start = geometry.get('start', [0, 0])
            end = geometry.get('end', [1, 1])
            
            return (
                min(start[0], end[0]),
                min(start[1], end[1]),
                max(start[0], end[0]),
                max(start[1], end[1])
            )
        
        elif element_type == 'circle':
            center = element.get('center', [0, 0])
            radius = element.get('radius', 1)
            
            return (
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius
            )
        
        else:
            # Default bounds
            return (0, 0, 1, 1)
    
    def _combine_bounds(self, bounds_list: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        """Combine multiple bounding boxes"""
        if not bounds_list:
            return (0, 0, 1, 1)
        
        min_x = min(bounds[0] for bounds in bounds_list)
        min_y = min(bounds[1] for bounds in bounds_list)
        max_x = max(bounds[2] for bounds in bounds_list)
        max_y = max(bounds[3] for bounds in bounds_list)
        
        return (min_x, min_y, max_x, max_y)
    
    def _calculate_scale_factor(self, bounds: Tuple[float, float, float, float]) -> float:
        """Calculate scale factor to fit bounds in image"""
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        if width == 0 or height == 0:
            return 1.0
        
        available_width = self.image_size[0] - 2 * self.padding
        available_height = self.image_size[1] - 2 * self.padding
        
        scale_x = available_width / width
        scale_y = available_height / height
        
        return min(scale_x, scale_y)
    
    def _calculate_offset(self, bounds: Tuple[float, float, float, float], scale_factor: float) -> Tuple[float, float]:
        """Calculate offset to center the element"""
        scaled_width = (bounds[2] - bounds[0]) * scale_factor
        scaled_height = (bounds[3] - bounds[1]) * scale_factor
        
        offset_x = (self.image_size[0] - scaled_width) / 2 - bounds[0] * scale_factor
        offset_y = (self.image_size[1] - scaled_height) / 2 - bounds[1] * scale_factor
        
        return (offset_x, offset_y)
    
    def _draw_element(self, img: np.ndarray, element: Dict[str, Any], 
                     scale_factor: float, offset: Tuple[float, float], 
                     color: Tuple[int, int, int], thickness: int):
        """Draw an element on the image"""
        element_type = element.get('type', '')
        
        if element_type == 'rectangle':
            self._draw_rectangle(img, element, scale_factor, offset, color, thickness)
        elif element_type in ['polyline', 'polygon']:
            self._draw_polyline(img, element, scale_factor, offset, color, thickness)
        elif element_type == 'line':
            self._draw_line(img, element, scale_factor, offset, color, thickness)
        elif element_type == 'circle':
            self._draw_circle(img, element, scale_factor, offset, color, thickness)
    
    def _transform_point(self, point: List[float], scale_factor: float, offset: Tuple[float, float]) -> Tuple[int, int]:
        """Transform a point to image coordinates"""
        x = int(point[0] * scale_factor + offset[0])
        y = int(point[1] * scale_factor + offset[1])
        return (x, y)
    
    def _draw_rectangle(self, img: np.ndarray, element: Dict[str, Any], 
                       scale_factor: float, offset: Tuple[float, float], 
                       color: Tuple[int, int, int], thickness: int):
        """Draw rectangle on image"""
        center = element.get('center', [0, 0])
        width = element.get('width', 1)
        height = element.get('height', 1)
        
        top_left = [center[0] - width/2, center[1] - height/2]
        bottom_right = [center[0] + width/2, center[1] + height/2]
        
        pt1 = self._transform_point(top_left, scale_factor, offset)
        pt2 = self._transform_point(bottom_right, scale_factor, offset)
        
        cv2.rectangle(img, pt1, pt2, color, thickness)
    
    def _draw_polyline(self, img: np.ndarray, element: Dict[str, Any], 
                      scale_factor: float, offset: Tuple[float, float], 
                      color: Tuple[int, int, int], thickness: int):
        """Draw polyline on image"""
        points = element.get('points', [])
        if len(points) < 2:
            return
        
        transformed_points = [self._transform_point(p, scale_factor, offset) for p in points]
        pts = np.array(transformed_points, np.int32)
        
        if element.get('type') == 'polygon':
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], False, color, thickness)
    
    def _draw_line(self, img: np.ndarray, element: Dict[str, Any], 
                  scale_factor: float, offset: Tuple[float, float], 
                  color: Tuple[int, int, int], thickness: int):
        """Draw line on image"""
        geometry = element.get('geometry', {})
        start = geometry.get('start', [0, 0])
        end = geometry.get('end', [1, 1])
        
        pt1 = self._transform_point(start, scale_factor, offset)
        pt2 = self._transform_point(end, scale_factor, offset)
        
        cv2.line(img, pt1, pt2, color, thickness)
    
    def _draw_circle(self, img: np.ndarray, element: Dict[str, Any], 
                    scale_factor: float, offset: Tuple[float, float], 
                    color: Tuple[int, int, int], thickness: int):
        """Draw circle on image"""
        center = element.get('center', [0, 0])
        radius = element.get('radius', 1)
        
        center_pt = self._transform_point(center, scale_factor, offset)
        scaled_radius = int(radius * scale_factor)
        
        cv2.circle(img, center_pt, scaled_radius, color, thickness)

class DeepLearningClassifier:
    """Deep learning classifier for architectural elements"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['door', 'wall', 'window', 'room', 'kitchen', 'bathroom', 'unknown']
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.model = ArchitecturalCNN(num_classes=self.num_classes)
        self.model.to(self.device)
        
        # Load pre-trained weights
        self._load_pretrained_weights()
        
        # Image converter
        self.converter = VectorToImageConverter()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Deep learning classifier initialized on {self.device}")
    
    def _load_pretrained_weights(self):
        """Load pre-trained model weights"""
        try:
            model_weights = ModelWeights()
            weights = model_weights.get_cnn_weights()
            
            if weights:
                # Create a state dict with proper parameter names
                state_dict = {}
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i < len(weights):
                        # Reshape weights to match parameter shape
                        weight_tensor = torch.FloatTensor(weights[i])
                        if weight_tensor.shape == param.shape:
                            state_dict[name] = weight_tensor
                        else:
                            # Initialize with random weights if shape mismatch
                            state_dict[name] = param.data
                    else:
                        state_dict[name] = param.data
                
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pre-trained CNN weights")
            else:
                logger.warning("No pre-trained weights available, using random initialization")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {str(e)}")
            logger.info("Using randomly initialized weights")
    
    def classify_elements(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Classify a list of architectural elements"""
        self.model.eval()
        classifications = []
        
        # Limit elements for performance
        max_elements = 200
        elements_to_process = elements[:max_elements]
        if len(elements) > max_elements:
            logger.warning(f"Processing only first {max_elements} elements out of {len(elements)} for performance")
        
        with torch.no_grad():
            for i, element in enumerate(elements_to_process):
                try:
                    # Find nearby elements for context
                    context_elements = self._find_context_elements(element, elements, max_distance=5.0)
                    
                    # Convert to image
                    img = self.converter.convert_element_to_image(element, context_elements)
                    
                    # Preprocess image
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    outputs = self.model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    predicted_class = self.class_names[predicted.item()]
                    confidence = probabilities[0][predicted.item()].item()
                    
                    # Use rule-based fallback for low confidence predictions
                    if confidence < 0.6:
                        predicted_class = self._rule_based_fallback(element)
                    
                    classifications.append(predicted_class)
                    
                except Exception as e:
                    logger.warning(f"Error classifying element {i}: {str(e)}")
                    classifications.append('unknown')
        
        # Fill remaining elements with fallback classification if we processed fewer elements
        remaining_elements = elements[len(elements_to_process):]
        if remaining_elements:
            for element in remaining_elements:
                fallback_class = self._rule_based_fallback(element)
                classifications.append(fallback_class)
        
        logger.info(f"Deep learning classification completed for {len(elements)} elements")
        return classifications
    
    def _find_context_elements(self, target_element: Dict[str, Any], 
                             all_elements: List[Dict[str, Any]], 
                             max_distance: float) -> List[Dict[str, Any]]:
        """Find nearby elements to provide context"""
        target_center = target_element.get('center')
        if not target_center:
            return []
        
        context_elements = []
        
        for element in all_elements:
            if element is target_element:
                continue
            
            element_center = element.get('center')
            if element_center:
                distance = np.sqrt((target_center[0] - element_center[0])**2 + 
                                 (target_center[1] - element_center[1])**2)
                
                if distance <= max_distance:
                    context_elements.append(element)
        
        # Limit to closest 5 elements for performance
        if len(context_elements) > 5:
            distances = []
            for elem in context_elements:
                elem_center = elem.get('center', [0, 0])
                dist = np.sqrt((target_center[0] - elem_center[0])**2 + 
                             (target_center[1] - elem_center[1])**2)
                distances.append((dist, elem))
            
            distances.sort(key=lambda x: x[0])
            context_elements = [elem for _, elem in distances[:5]]
        
        return context_elements
    
    def _rule_based_fallback(self, element: Dict[str, Any]) -> str:
        """Simple rule-based fallback for low confidence predictions"""
        layer = element.get('layer', '').upper()
        element_type = element.get('type', '').lower()
        
        # Layer-based classification
        if any(keyword in layer for keyword in ['WALL', 'PART']):
            return 'wall'
        elif any(keyword in layer for keyword in ['DOOR', 'ENTR']):
            return 'door'
        elif any(keyword in layer for keyword in ['WIND', 'GLAZ']):
            return 'window'
        elif any(keyword in layer for keyword in ['ROOM', 'AREA']):
            return 'room'
        
        # Geometric classification for rectangles
        if element_type == 'rectangle':
            width = element.get('width', 0)
            height = element.get('height', 0)
            
            if width > 0 and height > 0:
                aspect_ratio = width / height
                area = width * height
                
                # Door heuristics
                if (0.6 <= width <= 1.5 and 1.8 <= height <= 2.5 and 
                    0.3 <= aspect_ratio <= 0.8):
                    return 'door'
                
                # Window heuristics
                elif (0.4 <= width <= 3.0 and 0.8 <= height <= 2.0 and 
                      0.8 <= aspect_ratio <= 4.0):
                    return 'window'
                
                # Room heuristics
                elif area > 2.0:
                    return 'room'
        
        return 'unknown'
    
    def get_prediction_confidence(self, element: Dict[str, Any]) -> float:
        """Get confidence score for a single element classification"""
        self.model.eval()
        
        try:
            # Convert to image
            img = self.converter.convert_element_to_image(element)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                max_prob = torch.max(probabilities).item()
            
            return max_prob
            
        except Exception as e:
            logger.warning(f"Error getting prediction confidence: {str(e)}")
            return 0.0
