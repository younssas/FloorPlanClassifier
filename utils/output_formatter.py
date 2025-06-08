import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

class ConstructionEstimatingFormatter:
    """Formats classification results for construction estimating with standardized JSON output"""
    
    def __init__(self):
        self.element_subtypes = {
            'wall': ['exterior', 'interior', 'load_bearing', 'partition'],
            'door': ['entry', 'interior', 'sliding', 'french', 'pocket'],
            'window': ['casement', 'double_hung', 'sliding', 'fixed', 'bay'],
            'room': ['bedroom', 'bathroom', 'kitchen', 'living_room', 'dining_room', 'office', 'hallway', 'closet'],
            'fixture': ['toilet', 'sink', 'bathtub', 'shower', 'cabinet', 'appliance']
        }
    
    def format_classification_results(self, 
                                    elements: List[Dict[str, Any]], 
                                    classifications: List[str],
                                    confidence_scores: Optional[List[float]] = None,
                                    validation_details: Optional[List[Dict]] = None,
                                    approach_name: str = "Enhanced Rule-Based") -> Dict[str, Any]:
        """Format classification results into standardized JSON for construction estimating"""
        
        if confidence_scores is None:
            confidence_scores = [0.8] * len(classifications)
        
        if validation_details is None:
            validation_details = [{}] * len(classifications)
        
        classified_elements = []
        
        for i, (element, classification, confidence, details) in enumerate(
            zip(elements, classifications, confidence_scores, validation_details)
        ):
            if classification != 'unknown':
                formatted_element = self._format_single_element(
                    element, classification, confidence, details, i
                )
                classified_elements.append(formatted_element)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(classified_elements)
        
        return {
            "metadata": {
                "classification_approach": approach_name,
                "timestamp": datetime.now().isoformat(),
                "total_elements_processed": len(elements),
                "total_classified_elements": len(classified_elements),
                "classification_confidence_avg": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            },
            "summary_statistics": summary,
            "classified_elements": classified_elements,
            "validation_report": self._generate_validation_report(classified_elements)
        }
    
    def _format_single_element(self, 
                              element: Dict[str, Any], 
                              classification: str, 
                              confidence: float,
                              details: Dict[str, Any],
                              index: int) -> Dict[str, Any]:
        """Format a single element for construction estimating output"""
        
        # Generate UUID for tracking
        object_id = str(uuid.uuid4())
        
        # Determine subtype
        subtype = self._determine_subtype(element, classification, details)
        
        # Calculate geometry references
        geometry_refs = self._calculate_geometry_references(element)
        
        # Extract fusion entity IDs
        fusion_entity_ids = self._extract_fusion_entity_ids(element)
        
        return {
            "object_id": object_id,
            "fusion_entity_ids": fusion_entity_ids,
            "class": classification.title(),
            "subtype": subtype,
            "confidence": round(confidence, 3),
            "support": {
                "rules_matched": details.get('rules_matched', []),
                "key_features": details.get('features', {}),
                "classification_method": details.get('classification_method', 'rule_based_analysis')
            },
            "geometry_refs": geometry_refs,
            "construction_properties": self._extract_construction_properties(element, classification)
        }
    
    def _determine_subtype(self, element: Dict[str, Any], classification: str, details: Dict[str, Any]) -> str:
        """Determine specific subtype based on element properties and classification details"""
        
        features = details.get('features', {})
        
        if classification == 'room':
            # Check for specific room patterns
            text_content = element.get('text', '').lower()
            for room_type in ['bedroom', 'bathroom', 'kitchen', 'living', 'dining', 'office']:
                if room_type in text_content:
                    return room_type
            return 'general'
        
        elif classification == 'door':
            # Determine door type based on size and context
            width_ratio = features.get('width_ratio', 0)
            if width_ratio > 0.02:
                return 'entry'
            return 'interior'
        
        elif classification == 'window':
            # Determine window type based on aspect ratio
            aspect_ratio = features.get('aspect_ratio', 1.0)
            if aspect_ratio > 2.0:
                return 'sliding'
            elif aspect_ratio < 0.7:
                return 'fixed'
            return 'double_hung'
        
        elif classification == 'wall':
            # Determine wall type based on length and connections
            length_ratio = features.get('length_ratio', 0)
            if length_ratio > 0.1:
                return 'exterior'
            return 'interior'
        
        elif classification == 'fixture':
            # Determine fixture type based on room context
            room_context = features.get('room_context', '')
            if 'bathroom' in room_context:
                return 'bathroom_fixture'
            elif 'kitchen' in room_context:
                return 'kitchen_fixture'
            return 'general_fixture'
        
        return 'standard'
    
    def _calculate_geometry_references(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate geometry references for construction estimating"""
        
        geometry_refs = {}
        
        # Bounding box in drawing coordinates
        if 'bbox' in element:
            geometry_refs['bbox_drawing_coords'] = element['bbox']
            geometry_refs['centroid'] = [
                (element['bbox'][0] + element['bbox'][2]) / 2,
                (element['bbox'][1] + element['bbox'][3]) / 2
            ]
        elif 'center' in element:
            geometry_refs['centroid'] = element['center']
            # Estimate bbox for circular elements
            if 'radius' in element:
                r = element['radius']
                cx, cy = element['center']
                geometry_refs['bbox_drawing_coords'] = [cx - r, cy - r, cx + r, cy + r]
        
        # Additional geometric properties
        if 'points' in element:
            geometry_refs['point_count'] = len(element['points'])
            geometry_refs['polyline_length'] = self._calculate_polyline_length(element['points'])
        
        if 'radius' in element:
            geometry_refs['radius'] = element['radius']
            geometry_refs['area'] = 3.14159 * element['radius'] ** 2
        
        return geometry_refs
    
    def _extract_fusion_entity_ids(self, element: Dict[str, Any]) -> List[str]:
        """Extract source entity IDs for traceability"""
        ids = []
        
        if 'id' in element:
            ids.append(str(element['id']))
        
        if 'fusion_id' in element:
            ids.append(str(element['fusion_id']))
        
        if 'entity_id' in element:
            ids.append(str(element['entity_id']))
        
        return ids if ids else [f"element_{hash(str(element)) % 10000}"]
    
    def _extract_construction_properties(self, element: Dict[str, Any], classification: str) -> Dict[str, Any]:
        """Extract properties relevant for construction estimating"""
        
        properties = {}
        
        if classification == 'wall':
            if 'bbox' in element:
                bbox = element['bbox']
                length = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
                thickness = min(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
                properties.update({
                    'estimated_length_units': round(length, 2),
                    'estimated_thickness_units': round(thickness, 2),
                    'linear_feet_estimate': round(length * 0.1, 2)  # Rough conversion
                })
        
        elif classification in ['door', 'window']:
            if 'bbox' in element:
                bbox = element['bbox']
                width = abs(bbox[2] - bbox[0])
                height = abs(bbox[3] - bbox[1])
                properties.update({
                    'width_units': round(width, 2),
                    'height_units': round(height, 2),
                    'area_units': round(width * height, 2)
                })
        
        elif classification == 'room':
            if 'bbox' in element:
                bbox = element['bbox']
                area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                properties.update({
                    'floor_area_units': round(area, 2),
                    'estimated_square_feet': round(area * 0.01, 2)  # Rough conversion
                })
        
        elif classification == 'fixture':
            if 'radius' in element:
                properties.update({
                    'fixture_size': element['radius'],
                    'installation_area': round(3.14159 * element['radius'] ** 2, 2)
                })
        
        return properties
    
    def _calculate_polyline_length(self, points: List[List[float]]) -> float:
        """Calculate total length of polyline"""
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            total_length += (dx**2 + dy**2)**0.5
        
        return total_length
    
    def _generate_summary_statistics(self, classified_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for construction estimating"""
        
        class_counts = {}
        subtype_counts = {}
        total_area = 0.0
        total_linear_feet = 0.0
        
        for element in classified_elements:
            element_class = element['class']
            subtype = element['subtype']
            
            # Count by class
            class_counts[element_class] = class_counts.get(element_class, 0) + 1
            
            # Count by subtype
            subtype_key = f"{element_class}:{subtype}"
            subtype_counts[subtype_key] = subtype_counts.get(subtype_key, 0) + 1
            
            # Accumulate area and linear measurements
            construction_props = element.get('construction_properties', {})
            total_area += construction_props.get('floor_area_units', 0)
            total_area += construction_props.get('area_units', 0)
            total_linear_feet += construction_props.get('linear_feet_estimate', 0)
        
        return {
            "element_counts_by_class": class_counts,
            "element_counts_by_subtype": subtype_counts,
            "estimated_totals": {
                "total_floor_area_units": round(total_area, 2),
                "total_linear_feet": round(total_linear_feet, 2),
                "door_count": class_counts.get('Door', 0),
                "window_count": class_counts.get('Window', 0),
                "room_count": class_counts.get('Room', 0),
                "fixture_count": class_counts.get('Fixture', 0)
            }
        }
    
    def _generate_validation_report(self, classified_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate validation report for quality assurance"""
        
        high_confidence = sum(1 for e in classified_elements if e['confidence'] >= 0.8)
        medium_confidence = sum(1 for e in classified_elements if 0.6 <= e['confidence'] < 0.8)
        low_confidence = sum(1 for e in classified_elements if e['confidence'] < 0.6)
        
        # No architectural limits - accept all detected elements
        door_count = sum(1 for e in classified_elements if e['class'] == 'Door')
        window_count = sum(1 for e in classified_elements if e['class'] == 'Window')
        
        validation_flags = []
        # No validation flags for high counts
        
        return {
            "confidence_distribution": {
                "high_confidence": high_confidence,
                "medium_confidence": medium_confidence,
                "low_confidence": low_confidence
            },
            "validation_flags": validation_flags,
            "quality_score": round((high_confidence + medium_confidence * 0.7) / len(classified_elements), 3) if classified_elements else 0.0
        }
    
    def export_to_json_file(self, formatted_results: Dict[str, Any], filename: str = None) -> str:
        """Export formatted results to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classification_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        
        return filename
