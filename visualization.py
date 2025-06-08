import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import seaborn as sns
from matplotlib.colors import ListedColormap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloorPlanVisualizer:
    """Visualizes floor plans with classification overlays"""
    
    def __init__(self):
        # Define color scheme for different element types
        self.class_colors = {
            'door': '#8B4513',      # Brown
            'wall': '#2F4F4F',      # Dark slate gray
            'window': '#4169E1',    # Royal blue
            'room': '#F0F8FF',      # Alice blue (light fill)
            'kitchen': '#FF6347',   # Tomato
            'bathroom': '#20B2AA',  # Light sea green
            'bedroom': '#DDA0DD',   # Plum
            'living_room': '#90EE90', # Light green
            'dining_room': '#FFB6C1', # Light pink
            'office': '#F0E68C',    # Khaki
            'hallway': '#D3D3D3',   # Light gray
            'closet': '#CD853F',    # Peru
            'unknown': '#696969',   # Dim gray
            'fixture': '#FF1493'    # Deep pink
        }
        
        # Edge colors for better visibility
        self.edge_colors = {
            'door': '#654321',
            'wall': '#1C1C1C',
            'window': '#191970',
            'room': '#4682B4',
            'kitchen': '#DC143C',
            'bathroom': '#008B8B',
            'bedroom': '#9370DB',
            'living_room': '#228B22',
            'dining_room': '#C71585',
            'office': '#DAA520',
            'hallway': '#A9A9A9',
            'closet': '#A0522D',
            'unknown': '#2F2F2F',
            'fixture': '#B22222'
        }
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_floor_plan_visualization(self, elements: List[Dict[str, Any]], 
                                      classifications: List[str],
                                      title: str = "Floor Plan Classification",
                                      figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Create a comprehensive floor plan visualization"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Original elements
        self._plot_original_elements(ax1, elements)
        ax1.set_title("Original Floor Plan Elements")
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Classified elements
        self._plot_classified_elements(ax2, elements, classifications)
        ax2.set_title("Classified Elements")
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        self._add_classification_legend(fig, classifications)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_original_elements(self, ax: plt.Axes, elements: List[Dict[str, Any]]):
        """Plot original elements without classification colors"""
        bounds = self._calculate_bounds(elements)
        
        for element in elements:
            self._draw_element(ax, element, color='black', fill_color='lightgray', alpha=0.7)
        
        self._set_axis_bounds(ax, bounds)
    
    def _plot_classified_elements(self, ax: plt.Axes, elements: List[Dict[str, Any]], 
                                classifications: List[str]):
        """Plot elements with classification colors"""
        bounds = self._calculate_bounds(elements)
        
        # Group elements by classification for better rendering
        classification_groups = {}
        for i, (element, classification) in enumerate(zip(elements, classifications)):
            if classification not in classification_groups:
                classification_groups[classification] = []
            classification_groups[classification].append((element, i))
        
        # Draw elements by classification (rooms first, then walls, then openings)
        draw_order = ['room', 'kitchen', 'bathroom', 'bedroom', 'living_room', 
                     'dining_room', 'office', 'hallway', 'closet', 'wall', 
                     'door', 'window', 'fixture', 'unknown']
        
        for classification in draw_order:
            if classification in classification_groups:
                color = self.class_colors.get(classification, '#696969')
                edge_color = self.edge_colors.get(classification, '#000000')
                
                for element, idx in classification_groups[classification]:
                    alpha = 0.8 if classification == 'room' else 0.9
                    self._draw_element(ax, element, color=edge_color, 
                                     fill_color=color, alpha=alpha)
                    
                    # Add element ID labels for debugging
                    center = element.get('center')
                    if center and classification not in ['wall']:
                        ax.annotate(f"{idx}", xy=center, fontsize=8, 
                                  ha='center', va='center', 
                                  bbox=dict(boxstyle="round,pad=0.2", 
                                          facecolor='white', alpha=0.7))
        
        self._set_axis_bounds(ax, bounds)
    
    def _draw_element(self, ax: plt.Axes, element: Dict[str, Any], 
                     color: str, fill_color: str, alpha: float = 0.7):
        """Draw a single element on the plot"""
        element_type = element.get('type', '').lower()
        
        if element_type == 'rectangle':
            self._draw_rectangle(ax, element, color, fill_color, alpha)
        elif element_type in ['polyline', 'polygon']:
            self._draw_polyline(ax, element, color, fill_color, alpha)
        elif element_type == 'line':
            self._draw_line(ax, element, color, alpha)
        elif element_type == 'circle':
            self._draw_circle(ax, element, color, fill_color, alpha)
        elif element_type == 'text':
            self._draw_text(ax, element, color)
    
    def _draw_rectangle(self, ax: plt.Axes, element: Dict[str, Any], 
                       color: str, fill_color: str, alpha: float):
        """Draw rectangle element"""
        center = element.get('center', [0, 0])
        width = element.get('width', 1)
        height = element.get('height', 1)
        
        # Calculate bottom-left corner
        x = center[0] - width / 2
        y = center[1] - height / 2
        
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor=color, 
                               facecolor=fill_color, alpha=alpha)
        ax.add_patch(rect)
    
    def _draw_polyline(self, ax: plt.Axes, element: Dict[str, Any], 
                      color: str, fill_color: str, alpha: float):
        """Draw polyline or polygon element"""
        points = element.get('points', [])
        if len(points) < 2:
            return
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        if element.get('type') == 'polygon' or (len(points) > 2 and 
            np.allclose(points[0], points[-1], atol=0.1)):
            # Closed polygon
            polygon = patches.Polygon(points, linewidth=2, edgecolor=color,
                                    facecolor=fill_color, alpha=alpha)
            ax.add_patch(polygon)
        else:
            # Open polyline
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=alpha)
    
    def _draw_line(self, ax: plt.Axes, element: Dict[str, Any], 
                  color: str, alpha: float):
        """Draw line element"""
        geometry = element.get('geometry', {})
        start = geometry.get('start', [0, 0])
        end = geometry.get('end', [1, 1])
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               color=color, linewidth=2, alpha=alpha)
    
    def _draw_circle(self, ax: plt.Axes, element: Dict[str, Any], 
                    color: str, fill_color: str, alpha: float):
        """Draw circle element"""
        center = element.get('center', [0, 0])
        radius = element.get('radius', 1)
        
        circle = patches.Circle(center, radius, linewidth=2, edgecolor=color,
                              facecolor=fill_color, alpha=alpha)
        ax.add_patch(circle)
    
    def _draw_text(self, ax: plt.Axes, element: Dict[str, Any], color: str):
        """Draw text element"""
        center = element.get('center', [0, 0])
        text = element.get('text', '')
        font_size = element.get('font_size', 12)
        
        ax.text(center[0], center[1], text, fontsize=min(font_size, 10),
               ha='center', va='center', color=color, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def _calculate_bounds(self, elements: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """Calculate bounding box for all elements"""
        all_points = []
        
        for element in elements:
            element_type = element.get('type', '').lower()
            
            if element_type == 'rectangle':
                center = element.get('center', [0, 0])
                width = element.get('width', 1)
                height = element.get('height', 1)
                
                all_points.extend([
                    [center[0] - width/2, center[1] - height/2],
                    [center[0] + width/2, center[1] + height/2]
                ])
            
            elif element_type in ['polyline', 'polygon']:
                points = element.get('points', [])
                all_points.extend(points)
            
            elif element_type == 'line':
                geometry = element.get('geometry', {})
                start = geometry.get('start', [0, 0])
                end = geometry.get('end', [1, 1])
                all_points.extend([start, end])
            
            elif element_type == 'circle':
                center = element.get('center', [0, 0])
                radius = element.get('radius', 1)
                all_points.extend([
                    [center[0] - radius, center[1] - radius],
                    [center[0] + radius, center[1] + radius]
                ])
            
            elif element_type == 'text':
                center = element.get('center', [0, 0])
                all_points.append(center)
        
        if not all_points:
            return (0, 0, 10, 10)
        
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        margin = 1.0  # Add margin around the plot
        return (min(x_coords) - margin, min(y_coords) - margin,
                max(x_coords) + margin, max(y_coords) + margin)
    
    def _set_axis_bounds(self, ax: plt.Axes, bounds: Tuple[float, float, float, float]):
        """Set axis bounds with proper aspect ratio"""
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
    
    def _add_classification_legend(self, fig: plt.Figure, classifications: List[str]):
        """Add legend showing classification colors"""
        unique_classes = list(set(classifications))
        unique_classes.sort()
        
        # Create legend elements
        legend_elements = []
        for class_name in unique_classes:
            color = self.class_colors.get(class_name, '#696969')
            legend_elements.append(patches.Patch(color=color, label=class_name.replace('_', ' ').title()))
        
        # Add legend to the figure
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  title="Element Classifications", title_fontsize=12, fontsize=10)
    
    def create_classification_summary_chart(self, classifications: List[str], 
                                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Create a summary chart of classification results"""
        # Count classifications
        from collections import Counter
        class_counts = Counter(classifications)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pie chart
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = [self.class_colors.get(label, '#696969') for label in labels]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title("Classification Distribution")
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_title("Classification Counts")
        ax2.set_xlabel("Element Type")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_visualization(self, elements: List[Dict[str, Any]], 
                                      results_dict: Dict[str, List[str]],
                                      figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create visualization comparing multiple classification approaches"""
        n_approaches = len(results_dict)
        fig, axes = plt.subplots(1, n_approaches, figsize=figsize)
        
        if n_approaches == 1:
            axes = [axes]
        
        for i, (approach_name, classifications) in enumerate(results_dict.items()):
            self._plot_classified_elements(axes[i], elements, classifications)
            axes[i].set_title(f"{approach_name} Classification")
            axes[i].set_aspect('equal')
            axes[i].grid(True, alpha=0.3)
        
        # Add shared legend
        unique_classes = set()
        for classifications in results_dict.values():
            unique_classes.update(classifications)
        
        legend_elements = []
        for class_name in sorted(unique_classes):
            color = self.class_colors.get(class_name, '#696969')
            legend_elements.append(patches.Patch(color=color, label=class_name.replace('_', ' ').title()))
        
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.1, 0.5),
                  title="Classifications", title_fontsize=12, fontsize=10)
        
        plt.suptitle("Classification Approach Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_accuracy_visualization(self, performance_data: Dict[str, Dict], 
                                    figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create visualization of performance metrics"""
        if not performance_data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No performance data available", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        approaches = list(performance_data.keys())
        
        # Prepare data
        inference_times = []
        element_counts = []
        accuracies = []
        
        for approach in approaches:
            data = performance_data[approach]
            inference_times.append(data.get('avg_inference_time', 0))
            element_counts.append(data.get('total_elements', 0))
            accuracies.append(data.get('accuracy', 0) * 100 if 'accuracy' in data else 0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Inference time comparison
        bars1 = ax1.bar(approaches, inference_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title("Average Inference Time")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars1, inference_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Element count comparison
        bars2 = ax2.bar(approaches, element_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title("Elements Processed")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, element_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Accuracy comparison (if available)
        if any(acc > 0 for acc in accuracies):
            bars3 = ax3.bar(approaches, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax3.set_title("Classification Accuracy")
            ax3.set_ylabel("Accuracy (%)")
            ax3.set_ylim(0, 100)
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, acc in zip(bars3, accuracies):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{acc:.1f}%', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, "Accuracy data not available", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Classification Accuracy")
        
        # Performance efficiency (elements per second)
        efficiency = [count / time if time > 0 else 0 
                     for count, time in zip(element_counts, inference_times)]
        
        bars4 = ax4.bar(approaches, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title("Processing Efficiency")
        ax4.set_ylabel("Elements/Second")
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars4, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{eff:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
