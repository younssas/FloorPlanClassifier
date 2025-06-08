import math
import numpy as np
from typing import List, Tuple, Dict, Any

class GeometryUtils:
    """Utility class for geometric calculations and spatial analysis"""
    
    @staticmethod
    def calculate_distance(point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def calculate_bbox_area(bbox: List[float]) -> float:
        """Calculate area of bounding box [x1, y1, x2, y2]"""
        if len(bbox) != 4:
            return 0.0
        return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    
    @staticmethod
    def calculate_bbox_center(bbox: List[float]) -> List[float]:
        """Calculate center point of bounding box"""
        if len(bbox) != 4:
            return [0.0, 0.0]
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    @staticmethod
    def points_are_collinear(p1: List[float], p2: List[float], p3: List[float], tolerance: float = 1e-6) -> bool:
        """Check if three points are approximately collinear"""
        # Calculate cross product
        cross_product = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        return cross_product < tolerance
    
    @staticmethod
    def calculate_polygon_area(points: List[List[float]]) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    @staticmethod
    def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
        """Check if point is inside polygon using ray casting"""
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    @staticmethod
    def calculate_aspect_ratio(bbox: List[float]) -> float:
        """Calculate aspect ratio of bounding box"""
        if len(bbox) != 4:
            return 1.0
        
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        
        if height == 0:
            return float('inf')
        
        return width / height
    
    @staticmethod
    def bboxes_overlap(bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap"""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return False
        
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                   bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
    
    @staticmethod
    def calculate_bbox_overlap_area(bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlapping area between two bounding boxes"""
        if not GeometryUtils.bboxes_overlap(bbox1, bbox2):
            return 0.0
        
        overlap_x1 = max(bbox1[0], bbox2[0])
        overlap_y1 = max(bbox1[1], bbox2[1])
        overlap_x2 = min(bbox1[2], bbox2[2])
        overlap_y2 = min(bbox1[3], bbox2[3])
        
        return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    
    @staticmethod
    def normalize_coordinates(coords: List[List[float]], bounds: Dict[str, float]) -> List[List[float]]:
        """Normalize coordinates to [0, 1] range based on drawing bounds"""
        normalized = []
        width = bounds.get('width', 1.0)
        height = bounds.get('height', 1.0)
        min_x = bounds.get('min_x', 0.0)
        min_y = bounds.get('min_y', 0.0)
        
        for coord in coords:
            norm_x = (coord[0] - min_x) / width if width > 0 else 0.0
            norm_y = (coord[1] - min_y) / height if height > 0 else 0.0
            normalized.append([norm_x, norm_y])
        
        return normalized
    
    @staticmethod
    def calculate_polyline_length(points: List[List[float]]) -> float:
        """Calculate total length of polyline"""
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(points) - 1):
            total_length += GeometryUtils.calculate_distance(points[i], points[i + 1])
        
        return total_length
    
    @staticmethod
    def simplify_polyline(points: List[List[float]], tolerance: float = 1.0) -> List[List[float]]:
        """Simplify polyline using Douglas-Peucker algorithm"""
        if len(points) <= 2:
            return points
        
        def perpendicular_distance(point: List[float], line_start: List[float], line_end: List[float]) -> float:
            """Calculate perpendicular distance from point to line"""
            if line_start == line_end:
                return GeometryUtils.calculate_distance(point, line_start)
            
            A = line_end[0] - line_start[0]
            B = line_end[1] - line_start[1]
            C = line_start[0] * line_end[1] - line_end[0] * line_start[1]
            
            return abs(A * point[1] - B * point[0] + C) / math.sqrt(A * A + B * B)
        
        def douglas_peucker(points_list: List[List[float]], epsilon: float) -> List[List[float]]:
            if len(points_list) <= 2:
                return points_list
            
            # Find the point with maximum distance
            max_distance = 0.0
            max_index = 0
            
            for i in range(1, len(points_list) - 1):
                distance = perpendicular_distance(points_list[i], points_list[0], points_list[-1])
                if distance > max_distance:
                    max_distance = distance
                    max_index = i
            
            # If maximum distance is greater than epsilon, recursively simplify
            if max_distance > epsilon:
                left_part = douglas_peucker(points_list[:max_index + 1], epsilon)
                right_part = douglas_peucker(points_list[max_index:], epsilon)
                
                # Combine results (remove duplicate point)
                return left_part[:-1] + right_part
            else:
                return [points_list[0], points_list[-1]]
        
        return douglas_peucker(points, tolerance)
