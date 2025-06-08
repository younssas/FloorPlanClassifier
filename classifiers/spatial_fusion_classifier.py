import numpy as np
import logging
import math
from typing import List, Dict, Any, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialFusionClassifier:
    """Specialized classifier for spatial fusion JSON data with relationship analysis"""
    
    def __init__(self):
        # Classification rules based on spatial fusion entity patterns
        self.text_patterns = {
            'room': {
                'room': ['great room', 'master', 'bedroom', 'bed', 'suite', 'kitchen', 'office', 'garage', 'bathroom', 'bath', 'laundry', 'living', 'family', 'salon', 'lounge', 'utility', 'mud', 'wash', 'study', 'den', 'library', 'work', 'car', 'parking', 'closet', 'storage', 'wardrobe', 'hall', 'corridor', 'foyer', 'entry', 'balcony', 'patio', 'deck', 'terrace', 'nook', 'dining', 'pantry', 'powder', 'covered', 'porch']
            },
            'labels': {
                'window': ['w', 'window', 'win'],
                'door': ['d', 'door', 'dr'],
                'fixture': ['wc', 'sink', 'tub', 'shower']
            }
        }
        
        # Geometric thresholds based on drawing scale
        self.size_thresholds = {
            'door_width_range_pt': (20, 120),
            'window_width_range_pt': (15, 200),
            'wall_min_length_pt': 50,
            'fixture_size_range_pt': (10, 100)
        }
        
    def classify_elements(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Classify spatial fusion entities using relationship and geometric analysis"""
        classifications = ['unknown'] * len(entities)
        
        # Create entity lookup for relationship analysis
        entity_lookup = {entity['entity_id']: entity for entity in entities}
        
        # Classify each entity
        for i, entity in enumerate(entities):
            classification = self._classify_single_entity(entity, entity_lookup)
            classifications[i] = classification
        
        # Post-process to refine classifications using global context
        classifications = self._refine_classifications(entities, classifications, entity_lookup)
        
        logger.info(f"Spatial fusion classification completed for {len(entities)} entities")
        return classifications
    
    def _classify_single_entity(self, entity: Dict[str, Any], entity_lookup: Dict[str, Dict]) -> str:
        """Classify a single entity based on its properties and relationships"""
        
        primary_type = entity.get('primary_type', '').lower()
        attributes = entity.get('attributes', {})
        geometry = entity.get('geometry', {})
        relationships = entity.get('relationships', [])
        bbox = entity.get('bbox_drawing_coords', [])
        
        # Text-based classification - handle both Text and annotation types
        if primary_type in ['text', 'annotation']:
            return self._classify_text_entity(entity, entity_lookup)
        
        # DoorArc entities are clearly doors  
        if primary_type == 'doorarc':
            return 'door'
        
        # Polyline classification (walls, windows, doors) - this is the key for walls
        if primary_type == 'polyline':
            return self._classify_polyline_entity(entity, relationships, entity_lookup)
        
        # Arc classification (doors, fixtures)
        if primary_type == 'arc':
            return self._classify_arc_entity(entity, relationships, entity_lookup)
        
        # Circle classification (fixtures)
        if primary_type == 'circle':
            return self._classify_circle_entity(entity, relationships, entity_lookup)
        
        # Spline classification (fixtures, decorative)
        if primary_type == 'spline':
            return self._classify_spline_entity(entity, relationships, entity_lookup)
        
        return 'unknown'
    
    def _classify_text_entity(self, entity: Dict[str, Any], entity_lookup: Dict[str, Dict]) -> str:
        """Classify text entities as room labels or element labels"""
        
        attributes = entity.get('attributes', {})
        text_content = attributes.get('clean_text', '').lower().strip()
        normalized_text = attributes.get('normalized_text', '').lower().strip()
        relationships = entity.get('relationships', [])
        bbox = entity.get('bbox_drawing_coords', [])
        
        if not text_content:
            return 'unknown'
        
        # Dimension text (contains measurements)
        if any(char in text_content for char in ['"', "'", 'x', '-', '×']) and any(char.isdigit() for char in text_content):
            return 'dimension'
        
        # Single letter element labels - your data doesn't have W/D labels
        if len(text_content) == 1:
            return 'label'
        
        # Check for room type patterns
        for room_type, patterns in self.text_patterns['room'].items():
            for pattern in patterns:
                if pattern in text_content or pattern in normalized_text:
                    return room_type
        
        # Check for element labels
        for label_type, patterns in self.text_patterns['labels'].items():
            for pattern in patterns:
                if pattern in text_content or pattern in normalized_text:
                    return f'{label_type}_label'
        
        # Check if text is inside an enclosed area (likely room label)
        if self._is_text_in_room_context(entity, relationships, entity_lookup):
            return 'room'
        
        # Large text in open areas is likely room labels
        if len(bbox) >= 4:
            text_area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
            if text_area > 100:  # Large text area
                return 'room'
        
        return 'unknown'
    
    def _classify_polyline_entity(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> str:
        """Classify polyline entities as walls, windows, or doors"""
        
        bbox = entity.get('bbox_drawing_coords', [])
        geometry = entity.get('geometry', {})
        
        if len(bbox) < 4:
            return 'unknown'
        
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        length = max(width, height)
        thickness = min(width, height)
        aspect_ratio = length / (thickness + 1e-6)
        
        # Check for relationships with window/door labels first
        has_window_label = self._has_label_relationship(entity, relationships, entity_lookup, 'w')
        has_door_label = self._has_label_relationship(entity, relationships, entity_lookup, 'd')
        
        # Check for DoorArc relationships (strong door indicator)
        has_door_arc = self._has_door_arc_relationship(entity, relationships, entity_lookup)
        
        # Door detection with multiple criteria
        if (has_door_label or has_door_arc or 
            (self.size_thresholds['door_width_range_pt'][0] <= width <= self.size_thresholds['door_width_range_pt'][1] and
             aspect_ratio < 5 and self._is_near_wall(entity, relationships, entity_lookup))):
            return 'door'
        
        # Window detection with label relationship
        if (has_window_label or 
            (self.size_thresholds['window_width_range_pt'][0] <= width <= self.size_thresholds['window_width_range_pt'][1] and
             aspect_ratio < 8 and self._is_near_wall(entity, relationships, entity_lookup))):
            return 'window'
        
        # Wall detection - simplified criteria for spatial fusion data
        # Most walls in this data are shorter linear elements
        if (length >= 15 and aspect_ratio > 5):  # Lower threshold for this dataset
            return 'wall'
        
        # Very long elements are definitely walls
        if length >= 30:
            return 'wall'
        
        return 'unknown'
    
    def _classify_arc_entity(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> str:
        """Classify arc entities - often door swings or fixtures"""
        
        geometry = entity.get('geometry', {})
        bbox = entity.get('bbox_drawing_coords', [])
        
        if len(bbox) >= 4:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            
            # Door swing arcs
            if (self.size_thresholds['door_width_range_pt'][0] <= width <= self.size_thresholds['door_width_range_pt'][1] and
                self._is_near_wall(entity, relationships, entity_lookup)):
                return 'door'
        
        # Check for fixture context
        if self._is_in_fixture_context(entity, relationships, entity_lookup):
            return 'fixture'
        
        return 'unknown'
    
    def _classify_circle_entity(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> str:
        """Classify circle entities - typically fixtures"""
        
        bbox = entity.get('bbox_drawing_coords', [])
        
        if len(bbox) >= 4:
            diameter = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
            
            # Size-appropriate circles are likely fixtures
            if (self.size_thresholds['fixture_size_range_pt'][0] <= diameter <= 
                self.size_thresholds['fixture_size_range_pt'][1]):
                return 'fixture'
        
        return 'unknown'
    
    def _classify_spline_entity(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> str:
        """Classify spline entities - often fixtures or decorative elements"""
        
        bbox = entity.get('bbox_drawing_coords', [])
        
        if len(bbox) >= 4:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            size = max(width, height)
            
            # Fixture-sized splines
            if (self.size_thresholds['fixture_size_range_pt'][0] <= size <= 
                self.size_thresholds['fixture_size_range_pt'][1]):
                
                # Check if in bathroom/kitchen context
                if self._is_in_fixture_context(entity, relationships, entity_lookup):
                    return 'fixture'
        
        return 'unknown'
    
    def _has_wall_relationships(self, relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity has relationships typical of walls"""
        
        wall_indicators = 0
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity:
                target_type = target_entity.get('primary_type', '').lower()
                rel_type = rel.get('type', '').lower()
                
                # Walls often connect to other walls, doors, windows
                if (target_type in ['polyline', 'arc', 'doorarc'] and 
                    rel_type in ['touches', 'connects', 'intersects']):
                    wall_indicators += 1
        
        return wall_indicators >= 1
    
    def _is_near_wall(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity is positioned near a wall"""
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity:
                target_type = target_entity.get('primary_type', '').lower()
                rel_type = rel.get('type', '').lower()
                
                # Check for proximity to potential walls
                if (target_type == 'polyline' and 
                    rel_type in ['touches', 'near', 'intersects'] and
                    self._is_likely_wall(target_entity)):
                    return True
        
        return False
    
    def _is_likely_wall(self, entity: Dict[str, Any]) -> bool:
        """Quick check if entity is likely a wall based on geometry"""
        
        bbox = entity.get('bbox_drawing_coords', [])
        if len(bbox) < 4:
            return False
        
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        length = max(width, height)
        aspect_ratio = length / (min(width, height) + 1e-6)
        
        return length >= self.size_thresholds['wall_min_length_pt'] and aspect_ratio > 5
    
    def _is_in_fixture_context(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity is in a context suitable for fixtures"""
        
        # Look for nearby room labels indicating bathroom, kitchen, etc.
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity and target_entity.get('primary_type') == 'Text':
                text_content = target_entity.get('attributes', {}).get('clean_text', '').lower()
                
                # Check for room types that typically have fixtures
                fixture_rooms = ['bathroom', 'kitchen', 'laundry', 'utility']
                for room_type in fixture_rooms:
                    if room_type in text_content:
                        return True
        
        return False
    
    def _is_text_in_room_context(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if text is positioned as a room label"""
        
        # Text in the center of enclosed areas is likely room labels
        centroid = entity.get('centroid', [])
        if not centroid:
            return False
        
        # Check for relationships with enclosing elements
        enclosed_count = 0
        for rel in relationships:
            rel_type = rel.get('type', '').lower()
            if rel_type in ['inside', 'enclosed', 'within']:
                enclosed_count += 1
        
        return enclosed_count > 0
    
    def _refine_classifications(self, entities: List[Dict[str, Any]], classifications: List[str], entity_lookup: Dict[str, Dict]) -> List[str]:
        """Refine classifications using global context and relationships"""
        
        refined = classifications.copy()
        
        # Count classified elements
        counts = Counter(classifications)
        
        # Apply spatial reasoning
        for i, entity in enumerate(entities):
            current_class = classifications[i]
            
            # Refine door classifications using DoorArc relationships
            if current_class == 'unknown':
                if self._is_door_by_relationship(entity, entity_lookup):
                    refined[i] = 'door'
            
            # Refine window classifications using label relationships
            elif current_class == 'unknown':
                if self._is_window_by_label_relationship(entity, entity_lookup):
                    refined[i] = 'window'
        
        return refined
    
    def _is_door_by_relationship(self, entity: Dict[str, Any], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity should be classified as door based on relationships with DoorArc"""
        
        relationships = entity.get('relationships', [])
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity and target_entity.get('primary_type', '').lower() == 'doorarc':
                rel_type = rel.get('type', '').lower()
                if rel_type in ['touches', 'near', 'connects']:
                    return True
        
        return False
    
    def _is_window_by_label_relationship(self, entity: Dict[str, Any], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity should be classified as window based on label relationships"""
        
        relationships = entity.get('relationships', [])
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity and target_entity.get('primary_type') == 'Text':
                text_content = target_entity.get('attributes', {}).get('clean_text', '').lower()
                if text_content in ['w', 'window', 'win']:
                    return True
        
        return False
    
    def _has_label_relationship(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict], label_text: str) -> bool:
        """Check if entity has relationship with specific label text"""
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity and target_entity.get('primary_type') == 'Text':
                text_content = target_entity.get('attributes', {}).get('clean_text', '').lower()
                if text_content == label_text.lower():
                    return True
        
        return False
    
    def _has_door_arc_relationship(self, entity: Dict[str, Any], relationships: List[Dict], entity_lookup: Dict[str, Dict]) -> bool:
        """Check if entity has relationship with DoorArc entities"""
        
        for rel in relationships:
            target_id = rel.get('target_id')
            target_entity = entity_lookup.get(target_id)
            
            if target_entity and target_entity.get('primary_type', '').lower() == 'doorarc':
                rel_type = rel.get('type', '').lower()
                if rel_type in ['touches', 'near', 'connects', 'intersects']:
                    return True
        
        return False
