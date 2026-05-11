#!/usr/bin/env python3
"""
Ablation Study for TriageGuard Multi-Feature Detection System

Evaluates the contribution of each feature module by systematically removing
one feature at a time and measuring the impact on detection performance.

Usage:
    python ablation_study.py --test-data test_pairs.csv --output ablation_results.csv
"""

import argparse
import csv
import sys
import os
import time
from typing import Dict, List, Tuple

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    # Fallback implementations
    def precision_score(y_true, y_pred, zero_division=0):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division
    
    def recall_score(y_true, y_pred, zero_division=0):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division
    
    def f1_score(y_true, y_pred, zero_division=0):
        prec = precision_score(y_true, y_pred, zero_division)
        rec = recall_score(y_true, y_pred, zero_division)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division


class AblationStudy:
    """Run ablation study by removing one feature at a time"""
    
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        Initialize ablation study
        
        Args:
            thresholds: Dictionary of thresholds for each feature
        """
        self.feature_modules = ['icon', 'so', 'smaliopcode', 'apicall']
        
        # Default thresholds
        if thresholds is None:
            self.thresholds = {
                'icon': 0.6,
                'so': 0.85,
                'smaliopcode': 0.85,
                'apicall': 0.80
            }
        else:
            self.thresholds = thresholds
    
    def make_decision(self, scores: Dict[str, float], enabled_features: List[str]) -> bool:
        """
        Make similarity decision based on enabled features
        
        Args:
            scores: Dictionary of feature scores
            enabled_features: List of features to use
            
        Returns:
            Boolean decision (True = similar, False = different)
        """
        decisions = []
        
        for feature in enabled_features:
            if feature in scores and scores[feature] is not None:
                threshold = self.thresholds.get(feature, 0.5)
                decisions.append(scores[feature] >= threshold)
        
        if not decisions:
            return False
        
        # Majority voting
        positive_votes = sum(decisions)
        return positive_votes >= (len(decisions) / 2)
    
    def evaluate_feature_combination(self, test_data, enabled_features: List[str]) -> Dict:
        """
        Evaluate detection performance with specific feature combination
        
        Args:
            test_data: DataFrame or list of dicts with test pairs
            enabled_features: List of features to enable
            
        Returns:
            Dictionary with performance metrics
        """
        y_true = []
        y_pred = []
        total_time = 0.0
        
        # Handle both DataFrame and dict list
        if pd is not None and isinstance(test_data, pd.DataFrame):
            rows = test_data.to_dict('records')
        else:
            rows = test_data
        
        for row in rows:
            # Extract ground truth label
            true_label = int(row.get('label', row.get('ground_truth', 0)))
            
            # Extract feature scores
            scores = {}
            for feature in self.feature_modules:
                score_key = f'{feature}_score'
                if score_key in row:
                    scores[feature] = float(row[score_key]) if row[score_key] is not None else None
            
            # Make prediction
            start_time = time.time()
            prediction = self.make_decision(scores, enabled_features)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            y_true.append(true_label)
            y_pred.append(1 if prediction else 0)
        
        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Compute confusion matrix manually if sklearn not available
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_detection_time': total_time / len(rows) if len(rows) > 0 else 0,
            'total_time': total_time,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def run_ablation(self, test_data) -> List[Dict]:
        """
        Run full ablation study
        
        Args:
            test_data: Test dataset
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Baseline: All features
        print("Evaluating baseline (all features)...")
        baseline_metrics = self.evaluate_feature_combination(test_data, self.feature_modules)
        results.append({
            'Removed_Feature': 'None',
            'Enabled_Features': ','.join(self.feature_modules),
            'Precision': baseline_metrics['precision'],
            'Recall': baseline_metrics['recall'],
            'F1_Score': baseline_metrics['f1_score'],
            'Avg_Time_s': baseline_metrics['avg_detection_time'],
            'TP': baseline_metrics['tp'],
            'TN': baseline_metrics['tn'],
            'FP': baseline_metrics['fp'],
            'FN': baseline_metrics['fn']
        })
        
        # Ablation: Remove one feature at a time
        for removed_feature in self.feature_modules:
            enabled_features = [f for f in self.feature_modules if f != removed_feature]
            print(f"Evaluating without {removed_feature}...")
            
            metrics = self.evaluate_feature_combination(test_data, enabled_features)
            results.append({
                'Removed_Feature': removed_feature,
                'Enabled_Features': ','.join(enabled_features),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'Avg_Time_s': metrics['avg_detection_time'],
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            })
        
        return results


def load_test_data(csv_path: str):
    """Load test data from CSV file"""
    if not os.path.exists(csv_path):
        print(f"Error: Test data file not found: {csv_path}")
        sys.exit(1)
    
    if pd is not None:
        return pd.read_csv(csv_path)
    else:
        # Fallback: load as list of dicts
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)


def save_results(results: List[Dict], output_path: str):
    """Save results to CSV file"""
    if not results:
        print("Warning: No results to save")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_path}")


def print_results_table(results: List[Dict]):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    
    # Print header
    print(f"{'Removed':<12} {'Enabled Features':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time(s)':<10}")
    print("-"*100)
    
    # Print baseline
    baseline = results[0]
    print(f"{'None':<12} {baseline['Enabled_Features']:<30} "
          f"{baseline['Precision']:<12.4f} {baseline['Recall']:<12.4f} "
          f"{baseline['F1_Score']:<12.4f} {baseline['Avg_Time_s']:<10.4f}")
    print("-"*100)
    
    # Print ablation results
    for result in results[1:]:
        print(f"{result['Removed_Feature']:<12} {result['Enabled_Features']:<30} "
              f"{result['Precision']:<12.4f} {result['Recall']:<12.4f} "
              f"{result['F1_Score']:<12.4f} {result['Avg_Time_s']:<10.4f}")
    
    print("="*100)
    
    # Print impact analysis
    baseline_f1 = results[0]['F1_Score']
    print("\nFEATURE IMPACT ANALYSIS (ΔF1 = Baseline F1 - Ablated F1)")
    print("-"*60)
    print(f"{'Feature':<12} {'F1 Without':<15} {'ΔF1':<15} {'Impact':<15}")
    print("-"*60)
    
    for result in results[1:]:
        feature = result['Removed_Feature']
        f1 = result['F1_Score']
        delta = baseline_f1 - f1
        
        if delta > 0.05:
            impact = "High"
        elif delta > 0.02:
            impact = "Medium"
        else:
            impact = "Low"
        
        print(f"{feature:<12} {f1:<15.4f} {delta:+15.4f} {impact:<15}")
    
    print("-"*60)


def main():
    parser = argparse.ArgumentParser(
        description='Run ablation study for TriageGuard multi-feature detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ablation_study.py --test-data test_pairs.csv --output ablation_results.csv
  
  # With custom thresholds
  python ablation_study.py --test-data test_pairs.csv \\
      --icon-threshold 0.7 --so-threshold 0.9 \\
      --output results.csv
        """
    )
    
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data CSV (columns: apk1, apk2, label, *_score)')
    parser.add_argument('--output', type=str, default='ablation_results.csv',
                        help='Output CSV file for results (default: ablation_results.csv)')
    parser.add_argument('--icon-threshold', type=float, default=0.6,
                        help='Threshold for icon similarity (default: 0.6)')
    parser.add_argument('--so-threshold', type=float, default=0.85,
                        help='Threshold for SO similarity (default: 0.85)')
    parser.add_argument('--smaliopcode-threshold', type=float, default=0.85,
                        help='Threshold for SmaliOpcode similarity (default: 0.85)')
    parser.add_argument('--apicall-threshold', type=float, default=0.80,
                        help='Threshold for ApiCall similarity (default: 0.80)')
    
    args = parser.parse_args()
    
    # Set up thresholds
    thresholds = {
        'icon': args.icon_threshold,
        'so': args.so_threshold,
        'smaliopcode': args.smaliopcode_threshold,
        'apicall': args.apicall_threshold
    }
    
    print("="*100)
    print("TriageGuard Ablation Study")
    print("="*100)
    print(f"\nTest Data: {args.test_data}")
    print(f"Output: {args.output}")
    print(f"\nThresholds:")
    for feature, threshold in thresholds.items():
        print(f"  {feature}: {threshold}")
    print("="*100)
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    test_data = load_test_data(args.test_data)
    
    if pd is not None and isinstance(test_data, pd.DataFrame):
        print(f"Loaded {len(test_data)} test pairs")
    else:
        print(f"Loaded {len(test_data)} test pairs")
    
    # Run ablation study
    study = AblationStudy(thresholds)
    results = study.run_ablation(test_data)
    
    # Print results
    print_results_table(results)
    
    # Save results
    save_results(results, args.output)
    
    print("\nAblation study completed successfully!")


if __name__ == '__main__':
    main()
