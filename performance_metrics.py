import time
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and analyzes performance metrics for classification approaches"""
    
    def __init__(self):
        self.results = {}
        self.ground_truth = None  # For accuracy calculations when available
        
    def add_result(self, approach_name: str, classifications: List[str], 
                  inference_time: float, element_count: int):
        """Add classification results for an approach"""
        if approach_name not in self.results:
            self.results[approach_name] = {
                'classifications': [],
                'inference_times': [],
                'element_counts': [],
                'timestamps': []
            }
        
        self.results[approach_name]['classifications'].append(classifications)
        self.results[approach_name]['inference_times'].append(inference_time)
        self.results[approach_name]['element_counts'].append(element_count)
        self.results[approach_name]['timestamps'].append(time.time())
        
        logger.info(f"Added result for {approach_name}: {element_count} elements in {inference_time:.3f}s")
    
    def set_ground_truth(self, ground_truth: List[str]):
        """Set ground truth classifications for accuracy calculation"""
        self.ground_truth = ground_truth
        logger.info(f"Ground truth set with {len(ground_truth)} classifications")
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all approaches"""
        summary = {}
        
        for approach_name, data in self.results.items():
            if not data['classifications']:
                continue
            
            # Calculate basic metrics
            total_elements = sum(data['element_counts'])
            total_time = sum(data['inference_times'])
            avg_inference_time = np.mean(data['inference_times'])
            avg_elements_per_run = np.mean(data['element_counts'])
            
            # Calculate throughput
            throughput = total_elements / total_time if total_time > 0 else 0
            
            approach_summary = {
                'total_elements': total_elements,
                'total_inference_time': total_time,
                'avg_inference_time': avg_inference_time,
                'avg_elements_per_run': avg_elements_per_run,
                'throughput': throughput,
                'runs_count': len(data['classifications'])
            }
            
            # Calculate accuracy if ground truth is available
            if self.ground_truth and data['classifications']:
                latest_classifications = data['classifications'][-1]  # Use latest run
                accuracy = self._calculate_accuracy(latest_classifications, self.ground_truth)
                approach_summary['accuracy'] = accuracy
                
                # Additional accuracy metrics
                precision, recall, f1 = self._calculate_detailed_metrics(
                    latest_classifications, self.ground_truth
                )
                approach_summary['precision'] = precision
                approach_summary['recall'] = recall
                approach_summary['f1_score'] = f1
            
            # Performance category assessment
            approach_summary['performance_category'] = self._assess_performance(approach_summary)
            
            summary[approach_name] = approach_summary
        
        return summary
    
    def _calculate_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate classification accuracy"""
        if len(predictions) != len(ground_truth):
            logger.warning(f"Prediction length ({len(predictions)}) != ground truth length ({len(ground_truth)})")
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]
        
        if not predictions:
            return 0.0
        
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        return correct / len(predictions)
    
    def _calculate_detailed_metrics(self, predictions: List[str], 
                                  ground_truth: List[str]) -> tuple:
        """Calculate precision, recall, and F1 score"""
        if len(predictions) != len(ground_truth):
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]
        
        if not predictions:
            return 0.0, 0.0, 0.0
        
        # Get unique classes
        all_classes = list(set(predictions + ground_truth))
        
        # Calculate metrics for each class
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        
        for class_name in all_classes:
            # True positives, false positives, false negatives
            tp = sum(1 for p, gt in zip(predictions, ground_truth) 
                    if p == class_name and gt == class_name)
            fp = sum(1 for p, gt in zip(predictions, ground_truth) 
                    if p == class_name and gt != class_name)
            fn = sum(1 for p, gt in zip(predictions, ground_truth) 
                    if p != class_name and gt == class_name)
            
            # Calculate precision and recall for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_precision[class_name] = precision
            class_recall[class_name] = recall
            class_f1[class_name] = f1
        
        # Calculate macro averages
        avg_precision = np.mean(list(class_precision.values()))
        avg_recall = np.mean(list(class_recall.values()))
        avg_f1 = np.mean(list(class_f1.values()))
        
        return avg_precision, avg_recall, avg_f1
    
    def _assess_performance(self, metrics: Dict[str, Any]) -> str:
        """Assess overall performance category"""
        avg_time = metrics.get('avg_inference_time', float('inf'))
        throughput = metrics.get('throughput', 0)
        accuracy = metrics.get('accuracy', 0)
        
        # Define performance thresholds
        time_thresholds = {'excellent': 0.1, 'good': 0.5, 'acceptable': 2.0}
        throughput_thresholds = {'excellent': 10, 'good': 5, 'acceptable': 1}
        accuracy_thresholds = {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.7}
        
        # Score each metric
        time_score = 0
        if avg_time <= time_thresholds['excellent']:
            time_score = 3
        elif avg_time <= time_thresholds['good']:
            time_score = 2
        elif avg_time <= time_thresholds['acceptable']:
            time_score = 1
        
        throughput_score = 0
        if throughput >= throughput_thresholds['excellent']:
            throughput_score = 3
        elif throughput >= throughput_thresholds['good']:
            throughput_score = 2
        elif throughput >= throughput_thresholds['acceptable']:
            throughput_score = 1
        
        accuracy_score = 0
        if accuracy >= accuracy_thresholds['excellent']:
            accuracy_score = 3
        elif accuracy >= accuracy_thresholds['good']:
            accuracy_score = 2
        elif accuracy >= accuracy_thresholds['acceptable']:
            accuracy_score = 1
        
        # Overall assessment
        total_score = time_score + throughput_score + accuracy_score
        
        if total_score >= 8:
            return 'excellent'
        elif total_score >= 6:
            return 'good'
        elif total_score >= 3:
            return 'acceptable'
        else:
            return 'poor'
    
    def compare_approaches(self) -> Dict[str, Any]:
        """Compare all approaches and identify the best performer"""
        if len(self.results) < 2:
            return {"message": "Need at least 2 approaches to compare"}
        
        summary = self.get_summary()
        comparison = {
            'fastest': None,
            'most_accurate': None,
            'most_efficient': None,
            'best_overall': None,
            'detailed_comparison': {}
        }
        
        # Find best in each category
        fastest_time = float('inf')
        highest_accuracy = 0.0
        highest_throughput = 0.0
        
        for approach, metrics in summary.items():
            avg_time = metrics.get('avg_inference_time', float('inf'))
            accuracy = metrics.get('accuracy', 0.0)
            throughput = metrics.get('throughput', 0.0)
            
            if avg_time < fastest_time:
                fastest_time = avg_time
                comparison['fastest'] = approach
            
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                comparison['most_accurate'] = approach
            
            if throughput > highest_throughput:
                highest_throughput = throughput
                comparison['most_efficient'] = approach
        
        # Calculate overall best (weighted score)
        best_score = 0.0
        for approach, metrics in summary.items():
            # Normalize metrics (higher is better)
            time_score = 1.0 / (metrics.get('avg_inference_time', 1.0) + 0.001)
            accuracy_score = metrics.get('accuracy', 0.0)
            throughput_score = metrics.get('throughput', 0.0) / 10.0  # Scale down
            
            # Weighted overall score
            overall_score = (0.3 * time_score + 0.5 * accuracy_score + 0.2 * throughput_score)
            
            if overall_score > best_score:
                best_score = overall_score
                comparison['best_overall'] = approach
            
            comparison['detailed_comparison'][approach] = {
                'time_score': time_score,
                'accuracy_score': accuracy_score,
                'throughput_score': throughput_score,
                'overall_score': overall_score
            }
        
        return comparison
    
    def get_classification_agreement(self) -> Dict[str, Any]:
        """Analyze agreement between different approaches"""
        if len(self.results) < 2:
            return {"message": "Need at least 2 approaches to analyze agreement"}
        
        approaches = list(self.results.keys())
        agreement_matrix = {}
        
        # Get latest classifications for each approach
        latest_classifications = {}
        for approach in approaches:
            if self.results[approach]['classifications']:
                latest_classifications[approach] = self.results[approach]['classifications'][-1]
        
        if len(latest_classifications) < 2:
            return {"message": "Not enough classification results to compare"}
        
        # Calculate pairwise agreement
        for i, approach1 in enumerate(approaches):
            for j, approach2 in enumerate(approaches[i+1:], i+1):
                if approach1 in latest_classifications and approach2 in latest_classifications:
                    class1 = latest_classifications[approach1]
                    class2 = latest_classifications[approach2]
                    
                    min_len = min(len(class1), len(class2))
                    agreements = sum(1 for k in range(min_len) if class1[k] == class2[k])
                    agreement_rate = agreements / min_len if min_len > 0 else 0.0
                    
                    pair_key = f"{approach1}_vs_{approach2}"
                    agreement_matrix[pair_key] = {
                        'agreement_rate': agreement_rate,
                        'agreements': agreements,
                        'total_compared': min_len
                    }
        
        # Find consensus classifications
        if len(latest_classifications) >= 3:
            consensus_classifications = self._find_consensus(latest_classifications)
            agreement_matrix['consensus'] = consensus_classifications
        
        return agreement_matrix
    
    def _find_consensus(self, classifications_dict: Dict[str, List[str]]) -> Dict[str, Any]:
        """Find consensus classifications across approaches"""
        approaches = list(classifications_dict.keys())
        all_classifications = list(classifications_dict.values())
        
        # Find minimum length
        min_len = min(len(classifications) for classifications in all_classifications)
        
        consensus = []
        confidence_scores = []
        
        for i in range(min_len):
            # Get all classifications for this element
            element_classifications = [classifications[i] for classifications in all_classifications]
            
            # Count votes
            vote_counts = Counter(element_classifications)
            most_common = vote_counts.most_common(1)[0]
            
            consensus_class = most_common[0]
            vote_count = most_common[1]
            confidence = vote_count / len(element_classifications)
            
            consensus.append(consensus_class)
            confidence_scores.append(confidence)
        
        return {
            'consensus_classifications': consensus,
            'confidence_scores': confidence_scores,
            'avg_confidence': np.mean(confidence_scores),
            'elements_with_full_agreement': sum(1 for score in confidence_scores if score == 1.0)
        }
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        summary = self.get_summary()
        comparison = self.compare_approaches()
        agreement = self.get_classification_agreement()
        
        report = ["=== ARCHITECTURAL ELEMENT CLASSIFICATION PERFORMANCE REPORT ===\n"]
        
        # Individual approach performance
        report.append("1. INDIVIDUAL APPROACH PERFORMANCE:")
        for approach, metrics in summary.items():
            report.append(f"\n{approach.upper()}:")
            report.append(f"  • Elements processed: {metrics['total_elements']}")
            report.append(f"  • Average inference time: {metrics['avg_inference_time']:.3f}s")
            report.append(f"  • Throughput: {metrics['throughput']:.1f} elements/second")
            
            if 'accuracy' in metrics:
                report.append(f"  • Accuracy: {metrics['accuracy']:.2%}")
                report.append(f"  • Precision: {metrics.get('precision', 0):.2%}")
                report.append(f"  • Recall: {metrics.get('recall', 0):.2%}")
                report.append(f"  • F1-score: {metrics.get('f1_score', 0):.2%}")
            
            report.append(f"  • Performance category: {metrics['performance_category']}")
        
        # Comparison results
        if 'fastest' in comparison:
            report.append(f"\n2. COMPARISON RESULTS:")
            report.append(f"  • Fastest approach: {comparison['fastest']}")
            if comparison.get('most_accurate'):
                report.append(f"  • Most accurate: {comparison['most_accurate']}")
            report.append(f"  • Most efficient: {comparison['most_efficient']}")
            report.append(f"  • Best overall: {comparison['best_overall']}")
        
        # Agreement analysis
        if 'message' not in agreement:
            report.append(f"\n3. APPROACH AGREEMENT:")
            for pair, data in agreement.items():
                if pair != 'consensus':
                    report.append(f"  • {pair}: {data['agreement_rate']:.2%} agreement "
                                f"({data['agreements']}/{data['total_compared']} elements)")
            
            if 'consensus' in agreement:
                consensus_data = agreement['consensus']
                report.append(f"  • Consensus confidence: {consensus_data['avg_confidence']:.2%}")
                report.append(f"  • Full agreement elements: {consensus_data['elements_with_full_agreement']}")
        
        return "\n".join(report)
    
    def export_results(self) -> Dict[str, Any]:
        """Export all results for external analysis"""
        return {
            'summary': self.get_summary(),
            'comparison': self.compare_approaches(),
            'agreement': self.get_classification_agreement(),
            'raw_results': self.results,
            'ground_truth': self.ground_truth,
            'report': self.generate_performance_report()
        }
