import json
import os
from typing import List, Dict, Any, Optional
import logging
import time
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClassifier:
    """LLM-based classifier using OpenAI API for contextual classification"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
        
        self.class_names = ['door', 'wall', 'window', 'room', 'kitchen', 'bathroom', 'bedroom', 
                           'living_room', 'dining_room', 'office', 'hallway', 'closet', 'unknown']
        
        self.batch_size = 20  # Larger batches for efficiency
        self.max_elements = 50   # Reduced for faster processing
        
        logger.info("LLM classifier initialized with OpenAI API")
    
    def classify_elements(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Classify a list of architectural elements using LLM"""
        classifications = []
        
        # Limit elements to process for performance
        elements_to_process = elements[:self.max_elements]
        if len(elements) > self.max_elements:
            logger.warning(f"Processing only first {self.max_elements} elements out of {len(elements)} for performance")
        
        # Process in batches to optimize API calls
        for i in range(0, len(elements_to_process), self.batch_size):
            batch = elements_to_process[i:i + self.batch_size]
            try:
                batch_classifications = self._classify_batch(batch, elements_to_process)
                classifications.extend(batch_classifications)
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                # Add fallback classifications for this batch
                fallback_batch = self._fallback_classification(batch)
                classifications.extend(fallback_batch)
        
        # Fill remaining elements with fallback classification if we processed fewer elements
        remaining_elements = elements[len(elements_to_process):]
        if remaining_elements:
            fallback_remaining = self._fallback_classification(remaining_elements)
            classifications.extend(fallback_remaining)
        
        logger.info(f"LLM classification completed for {len(elements)} elements")
        return classifications
    
    def _classify_batch(self, batch_elements: List[Dict[str, Any]], 
                       all_elements: List[Dict[str, Any]]) -> List[str]:
        """Classify a batch of elements efficiently"""
        try:
            # Create simplified element summaries
            element_summaries = []
            for i, element in enumerate(batch_elements):
                # Extract key features quickly
                primary_type = element.get('primary_type', 'unknown')
                bbox = element.get('bbox_drawing_coords', [])
                text_content = element.get('text_content', '').strip()
                
                # Calculate basic dimensions
                width = height = length = 0
                if len(bbox) >= 4:
                    width = abs(bbox[2] - bbox[0])
                    height = abs(bbox[3] - bbox[1])
                    length = max(width, height)
                
                summary = f"Element {i}: Type={primary_type}, Size={length:.1f}x{min(width,height):.1f}"
                if text_content:
                    summary += f", Text='{text_content}'"
                
                element_summaries.append(summary)
            
            # Create efficient prompt
            prompt = f"""Classify these architectural elements as one of: {', '.join(self.class_names)}

Elements to classify:
{chr(10).join(element_summaries)}

Rules:
- Polyline with length > 20 and high aspect ratio = wall
- Text with room names (bedroom, kitchen, etc.) = room
- Text with dimensions (numbers, quotes) = dimension
- DoorArc or door-related elements = door
- Window-related elements = window
- Small elements near walls = fixture

Return JSON: {{"classifications": ["class1", "class2", ...]}}"""
            
            # Single API call for the batch
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an architectural element classifier. Be concise and accurate."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")
            
            result = json.loads(content)
            classifications = result.get('classifications', [])
            
            # Ensure we have the right number of classifications
            while len(classifications) < len(batch_elements):
                classifications.append('unknown')
            
            return classifications[:len(batch_elements)]
            
        except Exception as e:
            logger.error(f"Error in LLM batch classification: {str(e)}")
            return ['unknown'] * len(batch_elements)
    
    def _generate_element_description(self, element: Dict[str, Any], 
                                    all_elements: List[Dict[str, Any]]) -> str:
        """Generate detailed description of an element for LLM"""
        description_parts = []
        
        # Basic properties
        element_type = element.get('type', 'unknown')
        element_id = element.get('id', 'unknown')
        layer = element.get('layer', 'none')
        
        description_parts.append(f"Element ID: {element_id}")
        description_parts.append(f"Type: {element_type}")
        description_parts.append(f"Layer: {layer}")
        
        # Geometric properties
        if element_type == 'rectangle':
            width = element.get('width', 0)
            height = element.get('height', 0)
            center = element.get('center', [0, 0])
            
            description_parts.append(f"Dimensions: {width:.2f}m × {height:.2f}m")
            description_parts.append(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
            
            if width > 0 and height > 0:
                area = width * height
                aspect_ratio = width / height
                description_parts.append(f"Area: {area:.2f} sq.m")
                description_parts.append(f"Aspect ratio: {aspect_ratio:.2f}")
        
        elif element_type in ['polyline', 'polygon']:
            points = element.get('points', [])
            description_parts.append(f"Number of points: {len(points)}")
            
            if len(points) >= 3:
                # Calculate approximate area for polygons
                area = self._calculate_polygon_area(points)
                description_parts.append(f"Approximate area: {area:.2f} sq.m")
        
        elif element_type == 'line':
            geometry = element.get('geometry', {})
            start = geometry.get('start', [0, 0])
            end = geometry.get('end', [0, 0])
            length = ((end[0] - start[0])**2 + (end[1] - start[1])**2)**0.5
            description_parts.append(f"Length: {length:.2f}m")
        
        elif element_type == 'circle':
            radius = element.get('radius', 0)
            center = element.get('center', [0, 0])
            area = 3.14159 * radius * radius
            description_parts.append(f"Radius: {radius:.2f}m")
            description_parts.append(f"Area: {area:.2f} sq.m")
            description_parts.append(f"Center: ({center[0]:.2f}, {center[1]:.2f})")
        
        # Text content if available
        if 'text' in element:
            description_parts.append(f"Text content: '{element['text']}'")
        
        # Spatial relationships
        nearby_elements = self._find_nearby_elements(element, all_elements)
        if nearby_elements:
            nearby_descriptions = []
            for nearby in nearby_elements[:3]:  # Limit to 3 nearest
                nearby_type = nearby.get('type', 'unknown')
                nearby_layer = nearby.get('layer', 'none')
                distance = self._calculate_distance(
                    element.get('center', [0, 0]),
                    nearby.get('center', [0, 0])
                )
                nearby_descriptions.append(f"{nearby_type} (layer: {nearby_layer}, {distance:.2f}m away)")
            
            description_parts.append(f"Nearby elements: {', '.join(nearby_descriptions)}")
        
        return "; ".join(description_parts)
    
    def _generate_context_description(self, all_elements: List[Dict[str, Any]]) -> str:
        """Generate overall context description of the floor plan"""
        context_parts = []
        
        # Count element types
        type_counts = {}
        layer_counts = {}
        
        for element in all_elements:
            elem_type = element.get('type', 'unknown')
            layer = element.get('layer', 'none')
            
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        context_parts.append(f"Total elements: {len(all_elements)}")
        
        # Element type distribution
        type_summary = [f"{count} {elem_type}s" for elem_type, count in type_counts.items()]
        context_parts.append(f"Element types: {', '.join(type_summary)}")
        
        # Layer distribution
        layer_summary = [f"{count} in {layer}" for layer, count in layer_counts.items() if layer != 'none']
        if layer_summary:
            context_parts.append(f"Layers: {', '.join(layer_summary)}")
        
        # Calculate overall bounds
        all_centers = [elem.get('center', [0, 0]) for elem in all_elements if elem.get('center')]
        if all_centers:
            x_coords = [center[0] for center in all_centers]
            y_coords = [center[1] for center in all_centers]
            bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            plan_width = bounds[2] - bounds[0]
            plan_height = bounds[3] - bounds[1]
            context_parts.append(f"Floor plan dimensions: {plan_width:.2f}m × {plan_height:.2f}m")
        
        return "; ".join(context_parts)
    
    def _create_batch_classification_prompt(self, element_descriptions: List[Dict],
                                          context_description: str) -> str:
        """Create prompt for batch classification"""
        prompt = f"""You are an expert architectural analyst. Given the following floor plan context and element descriptions, classify each architectural element.

FLOOR PLAN CONTEXT:
{context_description}

CLASSIFICATION GUIDELINES:
- door: Typically 0.6-1.5m wide, 1.8-2.5m tall, rectangular, often in walls, layers like A-DOOR
- window: Typically 0.4-3.0m wide, 0.8-2.0m tall, rectangular, wider than tall, layers like A-GLAZ
- wall: Linear elements, polylines, or thin rectangles, layers like A-WALL
- room: Large enclosed areas, polygons or large rectangles, >2 sq.m area
- kitchen: Rooms with kitchen-related text or layers, moderate size 5-50 sq.m
- bathroom: Rooms with bathroom-related text or layers, smaller size 2-20 sq.m
- bedroom: Rooms with bedroom-related text, moderate to large size
- living_room: Large rooms with living-related text
- dining_room: Rooms with dining-related text
- office: Rooms with office/study-related text
- hallway: Long narrow spaces connecting rooms
- closet: Small enclosed spaces, usually <3 sq.m

ELEMENTS TO CLASSIFY:
"""
        
        for elem_desc in element_descriptions:
            prompt += f"\nElement {elem_desc['index']}: {elem_desc['description']}"
        
        prompt += f"""

Provide your classification in JSON format:
{{
    "classifications": [
        {{"index": 0, "class": "class_name", "confidence": 0.95, "reasoning": "brief explanation"}},
        {{"index": 1, "class": "class_name", "confidence": 0.87, "reasoning": "brief explanation"}}
    ]
}}

Available classes: {', '.join(self.class_names)}
"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """You are an expert architectural analyst specializing in floor plan interpretation. 
Your task is to classify architectural elements based on their geometric properties, 
spatial relationships, layer information, and contextual clues. 

Always respond with valid JSON containing classifications, confidence scores (0-1), 
and brief reasoning for each classification. Consider both individual element properties 
and their relationships to nearby elements."""
    
    def _parse_batch_response(self, response: Dict[str, Any], expected_count: int) -> List[str]:
        """Parse LLM response and extract classifications"""
        try:
            classifications_data = response.get('classifications', [])
            classifications = ['unknown'] * expected_count
            
            for item in classifications_data:
                index = item.get('index')
                classification = item.get('class', 'unknown')
                
                if (isinstance(index, int) and 0 <= index < expected_count and 
                    classification in self.class_names):
                    classifications[index] = classification
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return ['unknown'] * expected_count
    
    def _fallback_classification(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Fallback classification using simple rules"""
        classifications = []
        
        for element in elements:
            layer = element.get('layer', '').upper()
            element_type = element.get('type', '').lower()
            
            # Simple layer-based classification
            if any(keyword in layer for keyword in ['WALL', 'PART']):
                classification = 'wall'
            elif any(keyword in layer for keyword in ['DOOR', 'ENTR']):
                classification = 'door'
            elif any(keyword in layer for keyword in ['WIND', 'GLAZ']):
                classification = 'window'
            elif any(keyword in layer for keyword in ['ROOM', 'AREA']):
                classification = 'room'
            elif element_type == 'rectangle':
                # Basic geometric classification
                width = element.get('width', 0)
                height = element.get('height', 0)
                
                if 0.6 <= width <= 1.5 and 1.8 <= height <= 2.5:
                    classification = 'door'
                elif 0.4 <= width <= 3.0 and 0.8 <= height <= 2.0:
                    classification = 'window'
                else:
                    classification = 'room'
            else:
                classification = 'unknown'
            
            classifications.append(classification)
        
        return classifications
    
    def _find_nearby_elements(self, target_element: Dict[str, Any], 
                            all_elements: List[Dict[str, Any]], 
                            max_distance: float = 3.0) -> List[Dict[str, Any]]:
        """Find nearby elements for spatial context"""
        target_center = target_element.get('center')
        if not target_center:
            return []
        
        nearby_elements = []
        
        for element in all_elements:
            if element is target_element:
                continue
            
            element_center = element.get('center')
            if element_center:
                distance = self._calculate_distance(target_center, element_center)
                if distance <= max_distance:
                    nearby_elements.append(element)
        
        # Sort by distance and return closest ones
        nearby_elements.sort(key=lambda x: self._calculate_distance(
            target_center, x.get('center', [0, 0])
        ))
        
        return nearby_elements[:5]  # Return up to 5 nearest elements
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def _calculate_polygon_area(self, points: List[List[float]]) -> float:
        """Calculate area of a polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def classify_single_element_with_reasoning(self, element: Dict[str, Any], 
                                             all_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify a single element and return detailed reasoning"""
        try:
            element_description = self._generate_element_description(element, all_elements)
            context_description = self._generate_context_description(all_elements)
            
            prompt = f"""Classify this single architectural element with detailed reasoning.

FLOOR PLAN CONTEXT:
{context_description}

ELEMENT TO CLASSIFY:
{element_description}

Provide classification with detailed reasoning in JSON format:
{{
    "classification": "class_name",
    "confidence": 0.95,
    "reasoning": "detailed explanation of why this classification was chosen",
    "alternative_classes": ["other_possible_class1", "other_possible_class2"],
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Available classes: {', '.join(self.class_names)}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")
            
            result = json.loads(content)
            return result
            
        except Exception as e:
            logger.error(f"Error in detailed classification: {str(e)}")
            return {
                "classification": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error occurred: {str(e)}",
                "alternative_classes": [],
                "key_factors": []
            }
