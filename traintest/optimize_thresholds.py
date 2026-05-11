import csv
try:
    import pandas as pd
except ImportError:
    pd = None
import numpy as np
import itertools
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
except ImportError:
    precision_score = None
    recall_score = None
    f1_score = None
    accuracy_score = None
    confusion_matrix = None
    classification_report = None
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class _SeriesView:
    def __init__(self, arr):
        self.values = arr

class _FrameView:
    def __init__(self, data):
        self._data = data

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __getitem__(self, key):
        return _SeriesView(self._data[key])

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    if pd is not None:
        df = pd.read_csv(csv_path)
        df['label'] = df['label'].astype(int)
        return df

    cols = {
        'label': [],
        'icon_sim': [],
        'so_sim': [],
        'smaliopcode_sim': [],
        'apicall_sim': [],
    }
    with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cols['label'].append(int(row['label']))
            cols['icon_sim'].append(float(row['icon_sim']))
            cols['so_sim'].append(float(row['so_sim']))
            cols['smaliopcode_sim'].append(float(row['smaliopcode_sim']))
            cols['apicall_sim'].append(float(row['apicall_sim']))
    data = {k: np.array(v, dtype=(np.int64 if k == 'label' else np.float64)) for k, v in cols.items()}
    return _FrameView(data)

def generate_threshold_pairs(start=0.0, end=1.0, step=0.1):
    """
    Generate (low, high) pairs where low <= high.
    """
    thresholds = np.arange(start, end + 1e-9, step)
    pairs = []
    for low in thresholds:
        for high in thresholds:
            if low <= high:
                pairs.append((round(low, 2), round(high, 2)))
    return pairs

def generate_single_thresholds(start=0.5, end=1.0, step=0.05):
    return [round(x, 2) for x in np.arange(start, end + 1e-9, step)]

def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0

def _confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp

def _binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, len(y_true))
    return precision, recall, f1, acc

def _per_class_metrics(y_true, y_pred, cls):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == cls) & (y_pred == cls)))
    fp = int(np.sum((y_true != cls) & (y_pred == cls)))
    fn = int(np.sum((y_true == cls) & (y_pred != cls)))
    support = int(np.sum(y_true == cls))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, support

def _classification_report_text(y_true, y_pred, digits=4):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p0, r0, f10, s0 = _per_class_metrics(y_true, y_pred, 0)
    p1, r1, f11, s1 = _per_class_metrics(y_true, y_pred, 1)
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0

    macro_p = (p0 + p1) / 2.0
    macro_r = (r0 + r1) / 2.0
    macro_f1 = (f10 + f11) / 2.0

    total = s0 + s1
    weighted_p = _safe_div(p0 * s0 + p1 * s1, total)
    weighted_r = _safe_div(r0 * s0 + r1 * s1, total)
    weighted_f1 = _safe_div(f10 * s0 + f11 * s1, total)

    def fmt(x):
        return f"{x:.{digits}f}"

    lines = []
    lines.append("              precision    recall  f1-score   support")
    lines.append("")
    lines.append(f"{'0':>11} {fmt(p0):>10} {fmt(r0):>9} {fmt(f10):>9} {s0:>9}")
    lines.append(f"{'1':>11} {fmt(p1):>10} {fmt(r1):>9} {fmt(f11):>9} {s1:>9}")
    lines.append("")
    lines.append(f"{'accuracy':>11} {'':>10} {'':>9} {fmt(acc):>9} {total:>9}")
    lines.append(f"{'macro avg':>11} {fmt(macro_p):>10} {fmt(macro_r):>9} {fmt(macro_f1):>9} {total:>9}")
    lines.append(f"{'weighted avg':>11} {fmt(weighted_p):>10} {fmt(weighted_r):>9} {fmt(weighted_f1):>9} {total:>9}")
    return "\n".join(lines)

def _precision_score(y_true, y_pred, zero_division=0):
    p, _, _, _ = _binary_metrics(y_true, y_pred)
    return p

def _recall_score(y_true, y_pred, zero_division=0):
    _, r, _, _ = _binary_metrics(y_true, y_pred)
    return r

def _f1_score(y_true, y_pred, zero_division=0):
    _, _, f1, _ = _binary_metrics(y_true, y_pred)
    return f1

def _accuracy_score(y_true, y_pred):
    _, _, _, acc = _binary_metrics(y_true, y_pred)
    return acc

def _confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]], dtype=int)

def _classification_report(y_true, y_pred, digits=4):
    return _classification_report_text(y_true, y_pred, digits=digits)

if precision_score is None:
    precision_score = _precision_score
if recall_score is None:
    recall_score = _recall_score
if f1_score is None:
    f1_score = _f1_score
if accuracy_score is None:
    accuracy_score = _accuracy_score
if confusion_matrix is None:
    confusion_matrix = _confusion_matrix
if classification_report is None:
    classification_report = _classification_report

def apply_pipeline(df, icon_thr, so_thr, smaliopcode_thr, apicall_thr):
    """
    Apply the 4-stage pipeline.
    Order: Icon -> SO -> Smali Opcode (SmaliOpcode) -> ApiCall
    
    icon_thr: (low, high)
    so_thr: (low, high)
    smaliopcode_thr: (low, high)
    apicall_thr: single float
    
    Returns: predictions (numpy array)
    """
    n = len(df)
    preds = np.full(n, -1, dtype=int)
    
    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    smaliopcode_sim = df['smaliopcode_sim'].values
    apicall_sim = df['apicall_sim'].values
    
    low, high = icon_thr
    mask_high = (icon_sim >= high) & (icon_sim != -1)
    preds[mask_high] = 1
    
    mask_low = (icon_sim < low) & (icon_sim != -1)
    preds[mask_low] = 0
    
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    low, high = so_thr
    curr_sim = so_sim
    
    mask_high = undetermined & (curr_sim >= high) & (curr_sim != -1)
    preds[mask_high] = 1
    
    mask_low = undetermined & (curr_sim < low) & (curr_sim != -1)
    preds[mask_low] = 0
    
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    low, high = smaliopcode_thr
    curr_sim = smaliopcode_sim
    
    mask_high = undetermined & (curr_sim >= high) & (curr_sim != -1)
    preds[mask_high] = 1
    
    mask_low = undetermined & (curr_sim < low) & (curr_sim != -1)
    preds[mask_low] = 0
    
    undetermined = (preds == -1)
    if not np.any(undetermined):
        return preds
        
    curr_sim = apicall_sim
    threshold = apicall_thr
    
    mask_pos = undetermined & (curr_sim >= threshold)
    preds[mask_pos] = 1
    
    mask_neg = undetermined & (curr_sim < threshold)
    preds[mask_neg] = 0
    
    preds[preds == -1] = 0
    
    return preds

def apply_pipeline_negonly(df, icon_low, so_low, smaliopcode_thr, apicall_thr):
    n = len(df)
    preds = np.full(n, -1, dtype=int)

    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    smaliopcode_sim = df['smaliopcode_sim'].values
    apicall_sim = df['apicall_sim'].values

    valid = (icon_sim != -1)
    preds[valid & (icon_sim < icon_low)] = 0

    undetermined = (preds == -1)
    if not np.any(undetermined):
        preds[preds == -1] = 0
        return preds

    valid = (so_sim != -1)
    preds[undetermined & valid & (so_sim < so_low)] = 0

    undetermined = (preds == -1)
    if not np.any(undetermined):
        preds[preds == -1] = 0
        return preds

    low, high = smaliopcode_thr
    valid = (smaliopcode_sim != -1)
    preds[undetermined & valid & (smaliopcode_sim >= high)] = 1
    preds[undetermined & valid & (smaliopcode_sim < low)] = 0

    undetermined = (preds == -1)
    if not np.any(undetermined):
        preds[preds == -1] = 0
        return preds

    preds[undetermined & (apicall_sim >= apicall_thr)] = 1
    preds[undetermined & (apicall_sim < apicall_thr)] = 0
    preds[preds == -1] = 0
    return preds

def analyze_pipeline_performance_negonly(df, icon_low, so_low, smaliopcode_thr, apicall_thr):
    stats = []
    n_total = len(df)
    labels = df['label'].values
    preds = np.full(n_total, -1, dtype=int)

    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    smaliopcode_sim = df['smaliopcode_sim'].values
    apicall_sim = df['apicall_sim'].values

    stages = [
        ('Icon', ('negonly', icon_low), icon_sim),
        ('SO', ('negonly', so_low), so_sim),
        ('Smali Opcode', smaliopcode_thr, smaliopcode_sim),
        ('ApiCall', apicall_thr, apicall_sim)
    ]

    current_undetermined_mask = np.ones(n_total, dtype=bool)

    print("\nDetailed Layer Analysis:")
    print(f"{'Stage':<10} | {'Input':<8} | {'Decided':<8} | {'Passed':<8} | {'Filt%':<8} | {'Cum_F1':<8} | {'Cum_Pre':<8} | {'Cum_Rec':<8}")
    print("-" * 90)

    header = "Stage,Input,Decided,Passed,Filtered_Pct,Cum_Precision,Cum_Recall,Cum_F1,Cum_Accuracy\n"
    csv_rows = []

    for name, thr, data in stages:
        n_input = int(np.sum(current_undetermined_mask))
        newly_decided_mask = np.zeros(n_total, dtype=bool)

        description = ""

        if isinstance(thr, tuple) and len(thr) == 2 and thr[0] == 'negonly':
            low = float(thr[1])
            description = f"NegOnly Reject (Low={low})"
            valid_data_mask = (data != -1)
            mask_low = current_undetermined_mask & valid_data_mask & (data < low)
            preds[mask_low] = 0
            newly_decided_mask = mask_low
        elif name == 'Smali Opcode':
            low, high = thr
            description = f"Double Threshold (Low={low}, High={high})"
            valid_data_mask = (data != -1)
            mask_high = current_undetermined_mask & valid_data_mask & (data >= high)
            preds[mask_high] = 1
            mask_low = current_undetermined_mask & valid_data_mask & (data < low)
            preds[mask_low] = 0
            newly_decided_mask = mask_high | mask_low
        else:
            threshold = float(thr)
            description = f"Final Threshold (T={threshold})"
            mask_pos = current_undetermined_mask & (data >= threshold)
            preds[mask_pos] = 1
            mask_neg = current_undetermined_mask & (data < threshold)
            preds[mask_neg] = 0
            newly_decided_mask = mask_pos | mask_neg

        current_undetermined_mask = current_undetermined_mask & (~newly_decided_mask)

        n_decided = int(np.sum(newly_decided_mask))
        n_passed = int(np.sum(current_undetermined_mask))
        percent_filtered = (n_decided / n_total) * 100

        decided_indices = np.where(preds != -1)[0]
        if len(decided_indices) > 0:
            curr_preds = preds[decided_indices]
            curr_labels = labels[decided_indices]
            p = precision_score(curr_labels, curr_preds, zero_division=0)
            r = recall_score(curr_labels, curr_preds, zero_division=0)
            f1 = f1_score(curr_labels, curr_preds, zero_division=0)
            acc = accuracy_score(curr_labels, curr_preds)
        else:
            p, r, f1, acc = 0, 0, 0, 0

        print(f"{name:<10} | {n_input:<8} | {n_decided:<8} | {n_passed:<8} | {percent_filtered:<8.1f} | {f1:<8.4f} | {p:<8.4f} | {r:<8.4f}")

        stats.append({
            'Stage': name,
            'Input': n_input,
            'Decided': n_decided,
            'Passed': n_passed,
            'Filtered_Pct': percent_filtered,
            'Cum_Precision': p,
            'Cum_Recall': r,
            'Cum_F1': f1,
            'Cum_Accuracy': acc,
            'Description': description
        })

        csv_rows.append(f"{name},{n_input},{n_decided},{n_passed},{percent_filtered:.2f},{p:.4f},{r:.4f},{f1:.4f},{acc:.4f}")

    return stats, header + "\n".join(csv_rows)

def analyze_pipeline_performance(df, icon_thr, so_thr, smaliopcode_thr, apicall_thr):
    stats = []
    n_total = len(df)
    labels = df['label'].values
    
    preds = np.full(n_total, -1, dtype=int)
    
    icon_sim = df['icon_sim'].values
    so_sim = df['so_sim'].values
    smaliopcode_sim = df['smaliopcode_sim'].values
    apicall_sim = df['apicall_sim'].values
    
    stages = [
        ('Icon', icon_thr, icon_sim, True),
        ('SO', so_thr, so_sim, True),
        ('Smali Opcode', smaliopcode_thr, smaliopcode_sim, True),
        ('ApiCall', apicall_thr, apicall_sim, False)
    ]
    
    current_undetermined_mask = np.ones(n_total, dtype=bool)
    
    print("\nDetailed Layer Analysis:")
    print(f"{'Stage':<10} | {'Input':<8} | {'Decided':<8} | {'Passed':<8} | {'Filt%':<8} | {'Cum_F1':<8} | {'Cum_Pre':<8} | {'Cum_Rec':<8}")
    print("-" * 90)
    
    header = "Stage,Input,Decided,Passed,Filtered_Pct,Cum_Precision,Cum_Recall,Cum_F1,Cum_Accuracy\n"
    csv_rows = []
    
    for name, thr, data, is_double in stages:
        n_input = np.sum(current_undetermined_mask)
        
        newly_decided_mask = np.zeros(n_total, dtype=bool)
        
        description = ""
        
        if is_double:
            low, high = thr
            if low <= 0.0:
                description = f"High Threshold Only (High={high})"
            else:
                description = f"Double Threshold (Low={low}, High={high})"
                
            valid_data_mask = (data != -1)
            
            mask_high = current_undetermined_mask & valid_data_mask & (data >= high)
            preds[mask_high] = 1
            
            mask_low = current_undetermined_mask & valid_data_mask & (data < low)
            preds[mask_low] = 0
            
            newly_decided_mask = mask_high | mask_low
            
        else:
            threshold = thr
            description = f"Final Threshold (T={threshold})"
            
            mask_pos = current_undetermined_mask & (data >= threshold)
            preds[mask_pos] = 1
            
            mask_neg = current_undetermined_mask & (data < threshold)
            preds[mask_neg] = 0
            
            newly_decided_mask = mask_pos | mask_neg
        
        current_undetermined_mask = current_undetermined_mask & (~newly_decided_mask)
        
        n_decided = np.sum(newly_decided_mask)
        n_passed = np.sum(current_undetermined_mask)
        percent_filtered = (n_decided / n_total) * 100
        
        decided_indices = np.where(preds != -1)[0]
        
        if len(decided_indices) > 0:
            curr_preds = preds[decided_indices]
            curr_labels = labels[decided_indices]
            p = precision_score(curr_labels, curr_preds, zero_division=0)
            r = recall_score(curr_labels, curr_preds, zero_division=0)
            f1 = f1_score(curr_labels, curr_preds, zero_division=0)
            acc = accuracy_score(curr_labels, curr_preds)
        else:
            p, r, f1, acc = 0, 0, 0, 0
            
        print(f"{name:<10} | {n_input:<8} | {n_decided:<8} | {n_passed:<8} | {percent_filtered:<8.1f} | {f1:<8.4f} | {p:<8.4f} | {r:<8.4f}")
        
        stats.append({
            'Stage': name,
            'Input': n_input,
            'Decided': n_decided,
            'Passed': n_passed,
            'Filtered_Pct': percent_filtered,
            'Cum_Precision': p,
            'Cum_Recall': r,
            'Cum_F1': f1,
            'Cum_Accuracy': acc,
            'Description': description
        })
        
        csv_rows.append(f"{name},{n_input},{n_decided},{n_passed},{percent_filtered:.2f},{p:.4f},{r:.4f},{f1:.4f},{acc:.4f}")
        
    return stats, header + "\n".join(csv_rows)

plt = None
sns = None

def plot_sensitivity(df, best_config, output_dir):
    """
    Generate sensitivity plots for each layer around the best configuration.
    """
    global plt, sns
    if plt is None or sns is None:
        try:
            import matplotlib.pyplot as _plt
            import seaborn as _sns
            plt = _plt
            sns = _sns
        except ImportError:
            print("Skip plots: matplotlib/seaborn not available.")
            return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels = df['label'].values
    icon_p, so_p, smaliopcode_p, apicall_t = best_config
    
    plot_step = 0.05
    thresholds = np.arange(0.1, 1.0 + 1e-9, plot_step)
    
    print("\nGenerating sensitivity plots...")
    
    def plot_layer_metrics(layer_name, matrix_p, matrix_r, matrix_f1, thresholds, filename):
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        sns.heatmap(matrix_p, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[0])
        axes[0].set_title(f"{layer_name} - Precision")
        axes[0].set_xlabel("High Threshold")
        axes[0].set_ylabel("Low Threshold")
        
        sns.heatmap(matrix_r, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[1])
        axes[1].set_title(f"{layer_name} - Recall")
        axes[1].set_xlabel("High Threshold")
        axes[1].set_ylabel("Low Threshold")
        
        sns.heatmap(matrix_f1, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="viridis", annot=True, fmt=".2f", ax=axes[2])
        axes[2].set_title(f"{layer_name} - F1 Score")
        axes[2].set_xlabel("High Threshold")
        axes[2].set_ylabel("Low Threshold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    
    print("Plotting Icon sensitivity...")
    n = len(thresholds)
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, (low, high), so_p, smaliopcode_p, apicall_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("Icon Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_icon_metrics.png")

    print("Plotting SO sensitivity...")
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, icon_p, (low, high), smaliopcode_p, apicall_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("SO Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_so_metrics.png")

    print("Plotting Smali Opcode sensitivity...")
    p_matrix = np.zeros((n, n))
    r_matrix = np.zeros((n, n))
    f1_matrix = np.zeros((n, n))
    
    for i, low in enumerate(thresholds):
        for j, high in enumerate(thresholds):
            if low <= high:
                preds = apply_pipeline(df, icon_p, so_p, (low, high), apicall_t)
                p_matrix[i, j] = precision_score(labels, preds, zero_division=0)
                r_matrix[i, j] = recall_score(labels, preds, zero_division=0)
                f1_matrix[i, j] = f1_score(labels, preds, zero_division=0)
            else:
                p_matrix[i, j] = np.nan
                r_matrix[i, j] = np.nan
                f1_matrix[i, j] = np.nan
                
    plot_layer_metrics("Smali Opcode Layer", p_matrix, r_matrix, f1_matrix, thresholds, "sensitivity_smali_opcode_metrics.png")

    print("Plotting ApiCall sensitivity...")
    apicall_thresholds = np.arange(0.5, 1.0 + 1e-9, 0.02)
    f1_scores = []
    p_scores = []
    r_scores = []
    
    for thr in apicall_thresholds:
        preds = apply_pipeline(df, icon_p, so_p, smaliopcode_p, thr)
        f1_scores.append(f1_score(labels, preds, zero_division=0))
        p_scores.append(precision_score(labels, preds, zero_division=0))
        r_scores.append(recall_score(labels, preds, zero_division=0))
        
    plt.figure(figsize=(10, 6))
    plt.plot(apicall_thresholds, f1_scores, marker='o', label='F1 Score')
    plt.plot(apicall_thresholds, p_scores, marker='s', linestyle='--', label='Precision')
    plt.plot(apicall_thresholds, r_scores, marker='^', linestyle=':', label='Recall')
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("ApiCall Layer Sensitivity (Precision, Recall, F1)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sensitivity_apicall_metrics.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def plot_local_safety(df, output_dir):
    """
    Generate 'Local Safety' plots for each layer independently.
    Focus on:
    1. High Threshold Precision (Are we safe to say 'Malicious'?)
    2. Low Threshold NPV (Are we safe to say 'Benign'?)
    3. Coverage (How much data do we filter?)
    """
    global plt, sns
    if plt is None or sns is None:
        try:
            import matplotlib.pyplot as _plt
            import seaborn as _sns
            plt = _plt
            sns = _sns
        except ImportError:
            print("Skip plots: matplotlib/seaborn not available.")
            return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels = df['label'].values
    
    step = 0.05
    thresholds = np.arange(0.1, 1.0 + 1e-9, step)
    n = len(thresholds)
    
    layers = [
        ('Icon', df['icon_sim'].values),
        ('SO', df['so_sim'].values),
        ('Smali Opcode', df['smaliopcode_sim'].values)
    ]
    
    print("\nGenerating Local Safety plots (Independent Layer Analysis)...")
    
    for layer_name, data in layers:
        print(f"Plotting {layer_name} safety...")
        
        valid_mask = (data != -1)
        valid_data = data[valid_mask]
        valid_labels = labels[valid_mask]
        
        prec_high_matrix = np.zeros((n, n))
        npv_low_matrix = np.zeros((n, n))
        coverage_matrix = np.zeros((n, n))
        
        for i, low in enumerate(thresholds):
            for j, high in enumerate(thresholds):
                if low <= high:
                    mask_high = (valid_data >= high)
                    if np.sum(mask_high) > 0:
                        tp = np.sum((valid_labels[mask_high] == 1))
                        fp = np.sum((valid_labels[mask_high] == 0))
                        prec_high = tp / (tp + fp) if (tp + fp) > 0 else 0
                    else:
                        prec_high = 1.0
                        
                    mask_low = (valid_data < low)
                    if np.sum(mask_low) > 0:
                        tn = np.sum((valid_labels[mask_low] == 0))
                        fn = np.sum((valid_labels[mask_low] == 1))
                        npv_low = tn / (tn + fn) if (tn + fn) > 0 else 0
                    else:
                        npv_low = 1.0
                        
                    n_decided = np.sum(mask_high) + np.sum(mask_low)
                    coverage = n_decided / len(labels)
                    
                    prec_high_matrix[i, j] = prec_high
                    npv_low_matrix[i, j] = npv_low
                    coverage_matrix[i, j] = coverage
                    
                else:
                    prec_high_matrix[i, j] = np.nan
                    npv_low_matrix[i, j] = np.nan
                    coverage_matrix[i, j] = np.nan
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        sns.heatmap(prec_high_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=axes[0], vmin=0.8, vmax=1.0)
        axes[0].set_title(f"{layer_name} - High Threshold Precision\n(Safety of predicting 'Malicious')")
        axes[0].set_xlabel("High Threshold")
        axes[0].set_ylabel("Low Threshold")
        
        sns.heatmap(npv_low_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=axes[1], vmin=0.8, vmax=1.0)
        axes[1].set_title(f"{layer_name} - Low Threshold NPV\n(Safety of predicting 'Benign')")
        axes[1].set_xlabel("High Threshold")
        axes[1].set_ylabel("Low Threshold")
        
        sns.heatmap(coverage_matrix, xticklabels=[f"{x:.2f}" for x in thresholds], 
                    yticklabels=[f"{x:.2f}" for x in thresholds], 
                    cmap="Blues", annot=True, fmt=".2f", ax=axes[2])
        axes[2].set_title(f"{layer_name} - Filter Coverage %\n(Fraction of total data decided)")
        axes[2].set_xlabel("High Threshold")
        axes[2].set_ylabel("Low Threshold")
        
        plt.tight_layout()
        filename = f"safety_{layer_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def analyze_threshold_confidence(df, output_dir):
    """
    Generate confidence curves for picking High/Low thresholds independently.
    For each threshold T, calculate:
    1. Precision if we set High = T (samples >= T)
    2. NPV if we set Low = T (samples < T)
    """
    global plt
    if plt is None:
        try:
            import matplotlib.pyplot as _plt
            plt = _plt
        except ImportError:
            print("Skip plots: matplotlib not available.")
            return
    print("\nGenerating Threshold Confidence Analysis (Precision/NPV Curves)...")
    
    layers = [
        ('Icon', 'icon_sim'),
        ('SO', 'so_sim'),
        ('Smali Opcode', 'smaliopcode_sim')
    ]
    
    thresholds = np.arange(-0.2, 1.01, 0.01)
    labels = df['label'].values
    
    stats_all = []
    
    for name, col in layers:
        sim_scores = df[col].values
        
        n_total = len(sim_scores)
        n_missing = np.sum(sim_scores == -1)
        missing_ratio = n_missing / n_total * 100
        
        valid_mask = (sim_scores != -1)
        valid_scores = sim_scores[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_scores) == 0:
            print(f"Skipping {name} (No valid data)")
            continue
            
        min_val = valid_scores.min()
        max_val = valid_scores.max()
        
        print(f"Layer {name}: Valid Range [{min_val:.4f}, {max_val:.4f}], Missing: {missing_ratio:.1f}%")
        
        precisions = []
        npvs = []
        counts_high = []
        counts_low = []
        
        for t in thresholds:
            mask_high = (valid_scores >= t)
            n_high = np.sum(mask_high)
            
            if n_high > 0:
                tp = np.sum(valid_labels[mask_high] == 1)
                p = tp / n_high
            else:
                p = 1.0
            
            precisions.append(p)
            counts_high.append(n_high)
            
            mask_low = (valid_scores < t)
            n_low = np.sum(mask_low)
            
            if n_low > 0:
                tn = np.sum(valid_labels[mask_low] == 0)
                npv = tn / n_low
            else:
                npv = 1.0
            
            npvs.append(npv)
            counts_low.append(n_low)
            
            stats_all.append({
                'Layer': name,
                'Threshold': t,
                'Precision_Above': p,
                'NPV_Below': npv,
                'Count_Above': n_high,
                'Count_Below': n_low
            })
            
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        ax1.axvspan(min_val, max_val, color='gray', alpha=0.1, label='Valid Data Range')
        
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Confidence Score (Precision / NPV)', color='black')
        
        line1 = ax1.plot(thresholds, precisions, color='green', label='Precision (if High=T)', linewidth=2.5)
        
        line2 = ax1.plot(thresholds, npvs, color='red', label='NPV (if Low=T)', linewidth=2.5)
        
        ax1.axhline(0.95, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(0.98, color='gray', linestyle='--', alpha=0.5)
        ax1.text(thresholds[0], 0.955, '95% Conf.', color='gray', fontsize=8)
        
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, which='both', linestyle=':', alpha=0.6)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sample Count (Cumulative)', color='blue')
        
        line3 = ax2.plot(thresholds, counts_high, color='green', linestyle=':', alpha=0.3, label='Count >= T')
        ax2.fill_between(thresholds, counts_high, 0, color='green', alpha=0.05)
        
        line4 = ax2.plot(thresholds, counts_low, color='red', linestyle=':', alpha=0.3, label='Count < T')
        ax2.fill_between(thresholds, counts_low, 0, color='red', alpha=0.05)
        
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, len(valid_scores) * 1.1)
        
        plt.title(f"{name} Layer Analysis\nValid Data: {len(valid_scores)} ({100-missing_ratio:.1f}%) | Missing: {n_missing} ({missing_ratio:.1f}%)")
        
        lines = line1 + line2 + [line3[0], line4[0]]
        labs = [l.get_label() for l in lines]
        ax1.legend(lines, labs, loc='center right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confidence_curve_{name.lower().replace(' ', '_')}.png"))
        plt.close()
        
    out_csv = os.path.join(output_dir, 'layer_threshold_confidence.csv')
    if pd is not None:
        pd.DataFrame(stats_all).to_csv(out_csv, index=False)
    else:
        if stats_all:
            with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(stats_all[0].keys()))
                writer.writeheader()
                writer.writerows(stats_all)
        else:
            with open(out_csv, 'w', encoding='utf-8') as f:
                f.write("")
    print(f"Confidence stats saved to {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Optimize Multi-Stage Detection Thresholds")
    parser.add_argument('--csv', type=str, required=True, help='Path to results csv')
    parser.add_argument('--output', type=str, default='threshold_optimization_report.txt', help='Path to save the report')
    parser.add_argument('--plot_output', type=str, default=None, help='Directory to save plots (optional)')
    parser.add_argument('--mode', type=str, choices=['standard', 'negonly'], default='standard')
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--icon_low', type=float, default=None)
    parser.add_argument('--so_low', type=float, default=None)
    parser.add_argument('--smaliopcode_low', type=float, default=0.5)
    parser.add_argument('--smaliopcode_high', type=float, default=0.95)
    parser.add_argument('--apicall_thr', type=float, default=0.95)
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--min_recall', type=float, default=0.95)
    parser.add_argument('--target_precision', type=float, default=None)
    args = parser.parse_args()
    
    if args.plot_output is None:
        args.plot_output = os.path.dirname(os.path.abspath(args.output))

    
    print(f"Loading data from {args.csv}...")
    df = load_data(args.csv)
    labels = df['label'].values
    print(f"Loaded {len(df)} pairs. Positives: {sum(labels)}, Negatives: {len(labels)-sum(labels)}")

    best_config = None
    best_metrics = None
    preds = None
    stats = None
    csv_content = None
    early_rej = None
    early_fn = None

    if args.mode == 'negonly':
        smaliopcode_thr = (float(args.smaliopcode_low), float(args.smaliopcode_high))
        apicall_thr = float(args.apicall_thr)

        icon_sim = df['icon_sim'].values
        so_sim = df['so_sim'].values

        def eval_one(icon_low, so_low):
            nonlocal early_rej, early_fn
            preds_local = apply_pipeline_negonly(df, icon_low, so_low, smaliopcode_thr, apicall_thr)
            p_local = precision_score(labels, preds_local, zero_division=0)
            r_local = recall_score(labels, preds_local, zero_division=0)
            f1_local = f1_score(labels, preds_local, zero_division=0)

            m1 = (icon_sim != -1) & (icon_sim < icon_low)
            und = ~m1
            m2 = und & (so_sim != -1) & (so_sim < so_low)
            early_rej = int((m1 | m2).sum())
            early_fn = int(((labels == 1) & (m1 | m2)).sum())
            return preds_local, p_local, r_local, f1_local

        if args.icon_low is not None and args.so_low is not None:
            icon_low = float(args.icon_low)
            so_low = float(args.so_low)
            preds, p, r, f1 = eval_one(icon_low, so_low)
            best_config = (icon_low, so_low, smaliopcode_thr, apicall_thr)
            best_metrics = (p, r, f1)
        else:
            icon_lows = [round(x, 2) for x in np.arange(0.0, 0.5 + 1e-9, args.step)]
            so_lows = [round(x, 2) for x in np.arange(0.0, 1.0 + 1e-9, args.step)]

            best = None
            best_f1 = -1.0
            best_p = -1.0
            best_r = -1.0
            best_preds = None
            best_early_rej = None
            best_early_fn = None

            for icon_low in icon_lows:
                for so_low in so_lows:
                    preds_local, p_local, r_local, f1_local = eval_one(icon_low, so_low)

                    if args.min_recall is not None and r_local < args.min_recall:
                        continue
                    if args.target_precision is not None and p_local < args.target_precision:
                        continue

                    if f1_local > best_f1 or (abs(f1_local - best_f1) < 1e-12 and p_local > best_p):
                        best_f1 = f1_local
                        best_p = p_local
                        best_r = r_local
                        best = (icon_low, so_low)
                        best_preds = preds_local
                        best_early_rej = early_rej
                        best_early_fn = early_fn

            if best is None:
                for icon_low in icon_lows:
                    for so_low in so_lows:
                        preds_local, p_local, r_local, f1_local = eval_one(icon_low, so_low)
                        if f1_local > best_f1 or (abs(f1_local - best_f1) < 1e-12 and p_local > best_p):
                            best_f1 = f1_local
                            best_p = p_local
                            best_r = r_local
                            best = (icon_low, so_low)
                            best_preds = preds_local
                            best_early_rej = early_rej
                            best_early_fn = early_fn

            preds = best_preds
            early_rej = best_early_rej
            early_fn = best_early_fn
            best_config = (best[0], best[1], smaliopcode_thr, apicall_thr)
            best_metrics = (best_p, best_r, best_f1)

        icon_low, so_low, _, _ = best_config
        stats, csv_content = analyze_pipeline_performance_negonly(df, icon_low, so_low, smaliopcode_thr, apicall_thr)
    else:
        print("Starting Coordinate Descent Optimization...")

        current_config = {
            'icon': (0.2, 0.8),
            'smaliopcode': (0.2, 0.8),
            'so': (0.2, 0.8),
            'apicall': 0.8
        }

        print("Configuring Icon layer as Single Threshold (High Only)...")
        icon_space = [(-0.1, round(h, 2)) for h in np.arange(0.1, 1.0 + 1e-9, 0.05)]

        smaliopcode_space = generate_threshold_pairs(0.4, 1.0, 0.05)
        so_space = generate_threshold_pairs(0.4, 1.0, 0.05)
        apicall_space = generate_single_thresholds(0.5, 0.95, 0.05)

        max_cycles = 10

        for cycle in range(max_cycles):
            print(f"Cycle {cycle+1}...")
            changed = False

            best_local_f1 = -1
            best_local_val = current_config['icon']
            for val in icon_space:
                preds_local = apply_pipeline(df, val, current_config['so'], current_config['smaliopcode'], current_config['apicall'])
                f1_local = f1_score(labels, preds_local, zero_division=0)
                if f1_local > best_local_f1:
                    best_local_f1 = f1_local
                    best_local_val = val

            if best_local_val != current_config['icon']:
                print(f"  Icon updated: {current_config['icon']} -> {best_local_val} (F1: {best_local_f1:.4f})")
                current_config['icon'] = best_local_val
                changed = True

            best_local_f1 = -1
            best_local_val = current_config['so']
            for val in so_space:
                preds_local = apply_pipeline(df, current_config['icon'], val, current_config['smaliopcode'], current_config['apicall'])
                f1_local = f1_score(labels, preds_local, zero_division=0)
                if f1_local > best_local_f1:
                    best_local_f1 = f1_local
                    best_local_val = val

            if best_local_val != current_config['so']:
                print(f"  SO updated: {current_config['so']} -> {best_local_val} (F1: {best_local_f1:.4f})")
                current_config['so'] = best_local_val
                changed = True

            best_local_f1 = -1
            best_local_val = current_config['smaliopcode']
            for val in smaliopcode_space:
                preds_local = apply_pipeline(df, current_config['icon'], current_config['so'], val, current_config['apicall'])
                f1_local = f1_score(labels, preds_local, zero_division=0)
                if f1_local > best_local_f1:
                    best_local_f1 = f1_local
                    best_local_val = val

            if best_local_val != current_config['smaliopcode']:
                print(f"  Smali Opcode (SmaliOpcode) updated: {current_config['smaliopcode']} -> {best_local_val} (F1: {best_local_f1:.4f})")
                current_config['smaliopcode'] = best_local_val
                changed = True

            best_local_f1 = -1
            best_local_val = current_config['apicall']
            for val in apicall_space:
                preds_local = apply_pipeline(df, current_config['icon'], current_config['so'], current_config['smaliopcode'], val)
                f1_local = f1_score(labels, preds_local, zero_division=0)
                if f1_local > best_local_f1:
                    best_local_f1 = f1_local
                    best_local_val = val

            if best_local_val != current_config['apicall']:
                print(f"  ApiCall updated: {current_config['apicall']} -> {best_local_val} (F1: {best_local_f1:.4f})")
                current_config['apicall'] = best_local_val
                changed = True

            if not changed:
                print("Converged.")
                break

        preds = apply_pipeline(df, current_config['icon'], current_config['so'], current_config['smaliopcode'], current_config['apicall'])
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        best_config = (current_config['icon'], current_config['so'], current_config['smaliopcode'], current_config['apicall'])
        best_metrics = (p, r, f1)

    print("\n" + "="*50)
    print("Optimization Complete")
    print("="*50)
    
    p, r, f1 = best_metrics
    print(f"Best F1 Score: {f1:.4f}")
    print(f"Precision:     {p:.4f}")
    print(f"Recall:        {r:.4f}")
    print("-" * 30)
    print("Best Threshold Configuration:")
    if args.mode == 'negonly':
        icon_low, so_low, smaliopcode_thr, apicall_t = best_config
        print(f"Stage 1 (Icon): NegOnly Low={icon_low}")
        print(f"Stage 2 (SO):   NegOnly Low={so_low}")
        print(f"Stage 3 (Smali Opcode): Low={smaliopcode_thr[0]}, High={smaliopcode_thr[1]}")
        print(f"Stage 4 (ApiCall): Threshold={apicall_t}")
        print(f"Early Reject (Icon+SO): {early_rej} | Early FN: {early_fn}")
    else:
        icon_p, so_p, smaliopcode_p, apicall_t = best_config
        print(f"Stage 1 (Icon): Low={icon_p[0]}, High={icon_p[1]}")
        print(f"Stage 2 (SO):   Low={so_p[0]}, High={so_p[1]}")
        print(f"Stage 3 (Smali Opcode): Low={smaliopcode_p[0]}, High={smaliopcode_p[1]}")
        print(f"Stage 4 (ApiCall): Threshold={apicall_t}")
    print("-" * 30)
    
    if args.mode != 'negonly':
        stats, csv_content = analyze_pipeline_performance(df, icon_p, so_p, smaliopcode_p, apicall_t)
    
    cm = confusion_matrix(labels, preds)
    cr = classification_report(labels, preds, digits=4)
    
    tn, fp, fn, tp = cm.ravel()
    
    print("\nVerification - Confusion Matrix:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")
    print("\nClassification Report:")
    print(cr)
    
    with open(args.output, 'w') as f:
        f.write("Optimization Report\n")
        f.write("="*30 + "\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Best F1 Score: {f1:.4f}\n")
        f.write(f"Precision:     {p:.4f}\n")
        f.write(f"Recall:        {r:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Best Threshold Configuration:\n")
        if args.mode == 'negonly':
            icon_low, so_low, smaliopcode_thr, apicall_t = best_config
            f.write(f"Stage 1 (Icon): NegOnly Low={icon_low}\n")
            f.write(f"Stage 2 (SO):   NegOnly Low={so_low}\n")
            f.write(f"Stage 3 (Smali Opcode): Low={smaliopcode_thr[0]}, High={smaliopcode_thr[1]}\n")
            f.write(f"Stage 4 (ApiCall): Threshold={apicall_t}\n")
            f.write(f"Early Reject (Icon+SO): {early_rej} | Early FN: {early_fn}\n")
        else:
            f.write(f"Stage 1 (Icon): Low={icon_p[0]}, High={icon_p[1]}\n")
            f.write(f"Stage 2 (SO):   Low={so_p[0]}, High={so_p[1]}\n")
            f.write(f"Stage 3 (Smali Opcode): Low={smaliopcode_p[0]}, High={smaliopcode_p[1]}\n")
            f.write(f"Stage 4 (ApiCall): Threshold={apicall_t}\n")
        f.write("-" * 30 + "\n\n")
        
        f.write("Verification - Confusion Matrix:\n")
        f.write(f"TP: {tp} | FP: {fp}\n")
        f.write(f"FN: {fn} | TN: {tn}\n\n")
        f.write("Classification Report:\n")
        f.write(cr + "\n")
        
        f.write("Detailed Layer Analysis:\n")
        f.write(csv_content)
        
    print(f"\nReport saved to {args.output}")

    if pd is not None:
        excel_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'pipeline_step_metrics.xlsx')
        df_stats = pd.DataFrame(stats)
        df_stats = df_stats[['Stage', 'Input', 'Decided', 'Passed', 'Filtered_Pct', 'Cum_F1', 'Cum_Precision', 'Cum_Recall', 'Description']]
        df_stats.columns = [
            '层级 (Stage)',
            '输入数量 (Input)',
            '处理掉的数量 (Decided)',
            '剩余数量 (Passed)',
            '过滤比例% (Filtered)',
            '当前累计F1 (Cum F1)',
            '当前累计Precision (Cum Pre)',
            '当前累计Recall (Cum Rec)',
            '说明 (Description)'
        ]
        try:
            df_stats.to_excel(excel_path, index=False)
            print(f"Excel report saved to {excel_path}")
        except Exception as e:
            print(f"Skip Excel export: {e}")
    
    if (not args.no_plots) and args.mode != 'negonly':
        plot_sensitivity(df, best_config, args.plot_output)
        plot_local_safety(df, args.plot_output)
        analyze_threshold_confidence(df, args.plot_output)

if __name__ == '__main__':
    main()
