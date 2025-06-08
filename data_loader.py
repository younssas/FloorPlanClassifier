import fitz  # PyMuPDF
import json
import streamlit as st
from typing import Dict, List, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Handles PDF loading and vector extraction from floor plans"""
    
    def __init__(self):
        self.supported_elements = ['line', 'rect', 'circle', 'polyline', 'polygon']
    
    def extract_vectors(self, pdf_file) -> List[Dict[str, Any]]:
        """Extract vector data from PDF file"""
        try:
            # Read PDF file
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            elements = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Extract drawings (vector graphics)
                drawings = page.get_drawings()
                
                for i, drawing in enumerate(drawings):
                    element = self._process_drawing(drawing, page_num, i)
                    if element:
                        elements.append(element)
                
                # Extract text elements that might represent annotations
                text_instances = page.get_text("dict")
                for block in text_instances.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                if span.get("text", "").strip():
                                    text_element = self._process_text_element(span, page_num)
                                    if text_element:
                                        elements.append(text_element)
            
            pdf_document.close()
            logger.info(f"Extracted {len(elements)} elements from PDF")
            return elements
            
        except Exception as e:
            logger.error(f"Error extracting vectors from PDF: {str(e)}")
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def _process_drawing(self, drawing: Dict, page_num: int, element_id: int) -> Dict[str, Any]:
        """Process individual drawing element"""
        try:
            items = drawing.get("items", [])
            if not items:
                return None
            
            # Determine element type and extract geometry
            element = {
                "id": f"pdf_p{page_num}_e{element_id}",
                "page": page_num,
                "source": "pdf"
            }
            
            # Process different types of drawing items
            lines = []
            curves = []
            rects = []
            
            for item in items:
                if item[0] == "l":  # Line
                    lines.append({
                        "start": list(item[1]),
                        "end": list(item[2])
                    })
                elif item[0] == "re":  # Rectangle
                    rect = item[1]
                    rects.append({
                        "x": rect.x0,
                        "y": rect.y0,
                        "width": rect.width,
                        "height": rect.height
                    })
                elif item[0] == "c":  # Curve
                    curves.append({
                        "start": list(item[1]),
                        "control1": list(item[2]),
                        "control2": list(item[3]),
                        "end": list(item[4])
                    })
            
            # Classify element type based on geometry
            if rects:
                element.update({
                    "type": "rectangle",
                    "geometry": rects[0],
                    "width": rects[0]["width"],
                    "height": rects[0]["height"],
                    "center": [
                        rects[0]["x"] + rects[0]["width"] / 2,
                        rects[0]["y"] + rects[0]["height"] / 2
                    ]
                })
            elif len(lines) == 1:
                element.update({
                    "type": "line",
                    "geometry": lines[0],
                    "length": self._calculate_distance(lines[0]["start"], lines[0]["end"])
                })
            elif len(lines) > 1:
                # Multiple connected lines - polyline
                points = []
                for line in lines:
                    if not points:
                        points.extend([line["start"], line["end"]])
                    else:
                        points.append(line["end"])
                element.update({
                    "type": "polyline",
                    "points": points,
                    "geometry": {"lines": lines}
                })
            elif curves:
                element.update({
                    "type": "curve",
                    "geometry": curves[0]
                })
            else:
                return None
            
            # Extract additional properties
            if "fill" in drawing:
                element["fill"] = drawing["fill"]
            if "stroke" in drawing:
                element["stroke"] = drawing["stroke"]
            if "width" in drawing:
                element["stroke_width"] = drawing["width"]
            
            return element
            
        except Exception as e:
            logger.warning(f"Error processing drawing element: {str(e)}")
            return None
    
    def _process_text_element(self, span: Dict, page_num: int) -> Dict[str, Any]:
        """Process text elements that might be labels or annotations"""
        try:
            text = span.get("text", "").strip()
            if not text or len(text) < 2:
                return None
            
            bbox = span.get("bbox")
            if not bbox:
                return None
            
            return {
                "id": f"pdf_text_{page_num}_{hash(text) % 10000}",
                "type": "text",
                "text": text,
                "page": page_num,
                "source": "pdf",
                "bbox": list(bbox),
                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "font_size": span.get("size", 12)
            }
            
        except Exception as e:
            logger.warning(f"Error processing text element: {str(e)}")
            return None
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


class JSONProcessor:
    """Handles JSON metadata processing"""
    
    def __init__(self):
        self.required_fields = ['elements']
    
    def load_metadata(self, json_file) -> Dict[str, Any]:
        """Load and validate JSON metadata"""
        try:
            if hasattr(json_file, 'read'):
                json_file.seek(0)
                content = json_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                data = json.loads(content)
            else:
                with open(json_file, 'r') as f:
                    data = json.load(f)
            
            if not self._validate_structure(data):
                logger.error("Invalid JSON structure")
                return {"elements": []}
            
            # Handle spatial fusion format directly
            if 'entities' in data:
                logger.info(f"Processing spatial fusion format with {len(data['entities'])} entities")
                return data  # Return as-is for spatial fusion classifier
            
            # Handle standard format
            elements = []
            if 'elements' in data:
                for i, element in enumerate(data['elements']):
                    processed_element = self._process_element(element, i)
                    if processed_element:
                        elements.append(processed_element)
            
            logger.info(f"Processed {len(elements)} elements from JSON")
            return {"elements": elements}
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"elements": []}
        except Exception as e:
            logger.error(f"Error loading JSON metadata: {e}")
            return {"elements": []}
    
    def _validate_structure(self, data: Dict[str, Any]) -> bool:
        """Validate JSON structure"""
        if not isinstance(data, dict):
            return False
        
        # Accept spatial fusion format
        if 'entities' in data and isinstance(data['entities'], list):
            return True
        
        # Accept standard format    
        if 'elements' in data and isinstance(data['elements'], list):
            return True
            
        return False
    
    def _process_element(self, element: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process and standardize individual element"""
        try:
            # Handle enhanced format conversion
            if hasattr(self, '_convert_enhanced_format') and self._convert_enhanced_format:
                return self._convert_enhanced_element(element, index)
            
            # Handle simple format
            processed = {
                "id": element.get("id", f"json_elem_{index}"),
                "type": element.get("type", "unknown"),
                "source": "json"
            }
            
            # Copy all original fields
            processed.update(element)
            
            # Standardize geometry based on type
            if processed["type"] == "rectangle":
                if "center" in element and "width" in element and "height" in element:
                    center = element["center"]
                    width = element["width"]
                    height = element["height"]
                    processed.update({
                        "center": center,
                        "width": width,
                        "height": height,
                        "bbox": [
                            center[0] - width/2,
                            center[1] - height/2,
                            center[0] + width/2,
                            center[1] + height/2
                        ]
                    })
            
            elif processed["type"] == "polyline" and "points" in element:
                points = element["points"]
                processed["points"] = points
                if len(points) >= 2:
                    # Calculate bounding box
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    processed["bbox"] = [
                        min(x_coords), min(y_coords),
                        max(x_coords), max(y_coords)
                    ]
            
            elif processed["type"] == "circle":
                if "center" in element and "radius" in element:
                    center = element["center"]
                    radius = element["radius"]
                    processed.update({
                        "center": center,
                        "radius": radius,
                        "bbox": [
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius
                        ]
                    })
            
            # Extract layer information for rule-based classification
            if "layer" in element:
                processed["layer"] = element["layer"]
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error processing element {index}: {str(e)}")
            return None
    
    def _convert_enhanced_element(self, element: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Convert enhanced format element to standard format"""
        try:
            # Extract basic information
            processed = {
                "id": element.get("entity_id", f"enhanced_elem_{index}"),
                "type": element.get("primary_type", "unknown").lower(),
                "source": "enhanced_json"
            }
            
            # Extract centroid as center
            if "centroid" in element:
                processed["center"] = element["centroid"]
            
            # Extract bounding box
            if "bbox_drawing_coords" in element:
                bbox = element["bbox_drawing_coords"]
                processed["bbox"] = bbox
                
                # Calculate center if not available
                if "center" not in processed:
                    processed["center"] = [
                        (bbox[0] + bbox[2]) / 2,
                        (bbox[1] + bbox[3]) / 2
                    ]
                
                # Calculate dimensions for rectangles
                if processed["type"] in ["text", "rectangle"]:
                    processed["width"] = bbox[2] - bbox[0]
                    processed["height"] = bbox[3] - bbox[1]
            
            # Extract geometry information
            if "geometry" in element:
                geometry = element["geometry"]
                if "area" in geometry and geometry["area"]:
                    processed["area"] = geometry["area"]
                if "length" in geometry and geometry["length"]:
                    processed["length"] = geometry["length"]
                if "aspect_ratio" in geometry and geometry["aspect_ratio"]:
                    processed["aspect_ratio"] = geometry["aspect_ratio"]
            
            # Extract text content for text elements
            if processed["type"] == "text" and "text_content" in element:
                processed["text"] = element["text_content"]
            
            # Extract attributes
            if "attributes" in element:
                attrs = element["attributes"]
                
                # Layer information might be in different places
                if "layer" in attrs:
                    processed["layer"] = attrs["layer"]
                elif "text_category" in attrs:
                    processed["layer"] = f"A-{attrs['text_category'].upper()}"
                
                # Font information for text
                if "font" in attrs:
                    processed["font"] = attrs["font"]
                if "font_size" in attrs:
                    processed["font_size"] = attrs["font_size"]
            
            # Map enhanced types to standard types
            type_mapping = {
                "text": "text",
                "circle": "circle",
                "arc": "arc",
                "polyline": "polyline",
                "spline": "polyline",  # Treat splines as polylines
                "doorarc": "door"
            }
            
            if processed["type"] in type_mapping:
                processed["type"] = type_mapping[processed["type"]]
            
            # Infer architectural element type from relationships or attributes
            if "relationships" in element and element["relationships"]:
                # Analyze spatial relationships to improve classification
                nearby_types = [rel.get("target_primary_type", "") for rel in element["relationships"]]
                if "circle" in nearby_types and processed["type"] == "text":
                    # Text near circles might be labels
                    processed["layer"] = "A-ANNO"
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error converting enhanced element {index}: {str(e)}")
            return None


def merge_pdf_json_data(pdf_elements: List[Dict], json_elements: List[Dict]) -> List[Dict]:
    """Merge PDF and JSON data based on spatial proximity and IDs"""
    merged_elements = []
    used_pdf_indices = set()
    
    # First, try to match by ID
    for json_elem in json_elements:
        json_id = json_elem.get("id")
        matched = False
        
        for i, pdf_elem in enumerate(pdf_elements):
            if i in used_pdf_indices:
                continue
            
            pdf_id = pdf_elem.get("id")
            if json_id and pdf_id and json_id == pdf_id:
                # Direct ID match
                merged_elem = {**pdf_elem, **json_elem}
                merged_elem["source"] = "merged"
                merged_elements.append(merged_elem)
                used_pdf_indices.add(i)
                matched = True
                break
        
        if not matched:
            # Try spatial matching based on proximity
            json_center = json_elem.get("center")
            if json_center:
                best_match_idx = None
                best_distance = float('inf')
                
                for i, pdf_elem in enumerate(pdf_elements):
                    if i in used_pdf_indices:
                        continue
                    
                    pdf_center = pdf_elem.get("center")
                    if pdf_center:
                        distance = ((json_center[0] - pdf_center[0]) ** 2 + 
                                   (json_center[1] - pdf_center[1]) ** 2) ** 0.5
                        
                        if distance < best_distance and distance < 50:  # Proximity threshold
                            best_distance = distance
                            best_match_idx = i
                
                if best_match_idx is not None:
                    merged_elem = {**pdf_elements[best_match_idx], **json_elem}
                    merged_elem["source"] = "merged"
                    merged_elements.append(merged_elem)
                    used_pdf_indices.add(best_match_idx)
                    matched = True
        
        if not matched:
            # No PDF match found, use JSON element as-is
            merged_elements.append(json_elem)
    
    # Add remaining PDF elements that weren't matched
    for i, pdf_elem in enumerate(pdf_elements):
        if i not in used_pdf_indices:
            merged_elements.append(pdf_elem)
    
    logger.info(f"Merged data: {len(merged_elements)} total elements")
    return merged_elements
