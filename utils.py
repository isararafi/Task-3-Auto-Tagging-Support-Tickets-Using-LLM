import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_accuracy(predictions: List[List[Dict]], ground_truth: List[List[str]]) -> float:
    """Calculate accuracy based on top prediction matching ground truth."""
    correct = 0
    total = len(predictions)
    
    for pred, truth in zip(predictions, ground_truth):
        if pred and truth:
            # Check if top prediction is in ground truth
            top_prediction = pred[0]["category"] if pred else ""
            if top_prediction in truth:
                correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_precision_recall_f1(predictions: List[List[Dict]], ground_truth: List[List[str]]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, truth in zip(predictions, ground_truth):
        pred_categories = [p["category"] for p in pred] if pred else []
        truth_categories = truth if truth else []
        
        # Count true positives
        for pred_cat in pred_categories:
            if pred_cat in truth_categories:
                true_positives += 1
            else:
                false_positives += 1
        
        # Count false negatives
        for truth_cat in truth_categories:
            if truth_cat not in pred_categories:
                false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def create_confusion_matrix(predictions: List[List[Dict]], ground_truth: List[List[str]], categories: List[str]) -> pd.DataFrame:
    """Create a confusion matrix for multi-label classification."""
    matrix = pd.DataFrame(0, index=categories, columns=categories)
    
    for pred, truth in zip(predictions, ground_truth):
        pred_categories = [p["category"] for p in pred] if pred else []
        truth_categories = truth if truth else []
        
        for pred_cat in pred_categories:
            for truth_cat in truth_categories:
                if pred_cat in matrix.index and truth_cat in matrix.columns:
                    matrix.loc[pred_cat, truth_cat] += 1
    
    return matrix

def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of predicted categories."""
    # Count occurrences of each category
    category_counts = {}
    for i in range(1, 4):  # Top 3 predictions
        col = f"tag_{i}"
        if col in df.columns:
            counts = df[col].value_counts()
            for category, count in counts.items():
                if category:  # Skip empty values
                    category_counts[category] = category_counts.get(category, 0) + count
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(category_counts.keys()),
            y=list(category_counts.values()),
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Distribution of Predicted Categories",
        xaxis_title="Categories",
        yaxis_title="Count",
        xaxis_tickangle=-45
    )
    
    return fig

def plot_confidence_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of confidence scores."""
    confidence_scores = []
    for i in range(1, 4):  # Top 3 predictions
        col = f"confidence_{i}"
        if col in df.columns:
            confidence_scores.extend(df[col].dropna().tolist())
    
    fig = go.Figure(data=[
        go.Histogram(
            x=confidence_scores,
            nbinsx=20,
            marker_color='lightgreen'
        )
    ])
    
    fig.update_layout(
        title="Distribution of Confidence Scores",
        xaxis_title="Confidence Score",
        yaxis_title="Frequency"
    )
    
    return fig

def plot_performance_comparison(zero_shot_metrics: Dict, few_shot_metrics: Dict) -> go.Figure:
    """Plot comparison of zero-shot vs few-shot performance."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure(data=[
        go.Bar(
            name='Zero-shot',
            x=metrics,
            y=[zero_shot_metrics.get(metric, 0) for metric in metrics],
            marker_color='lightcoral'
        ),
        go.Bar(
            name='Few-shot',
            x=metrics,
            y=[few_shot_metrics.get(metric, 0) for metric in metrics],
            marker_color='lightblue'
        )
    ])
    
    fig.update_layout(
        title="Performance Comparison: Zero-shot vs Few-shot",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group'
    )
    
    return fig

def format_classification_results(classifications: List[List[Dict]]) -> str:
    """Format classification results for display."""
    if not classifications:
        return "No classifications available."
    
    result_text = ""
    for i, classification in enumerate(classifications):
        result_text += f"**Ticket {i+1}:**\n"
        if classification:
            for j, pred in enumerate(classification[:3]):  # Top 3
                category = pred.get("category", "Unknown")
                confidence = pred.get("confidence", 0.0)
                result_text += f"  {j+1}. {category} (Confidence: {confidence:.2f})\n"
        else:
            result_text += "  No classification available\n"
        result_text += "\n"
    
    return result_text

def create_summary_statistics(df: pd.DataFrame) -> Dict:
    """Create summary statistics for the classification results."""
    stats = {
        "total_tickets": len(df),
        "average_confidence": 0.0,
        "most_common_category": "",
        "categories_used": 0
    }
    
    # Calculate average confidence
    confidence_scores = []
    for i in range(1, 4):
        col = f"confidence_{i}"
        if col in df.columns:
            confidence_scores.extend(df[col].dropna().tolist())
    
    if confidence_scores:
        stats["average_confidence"] = np.mean(confidence_scores)
    
    # Find most common category
    category_counts = {}
    for i in range(1, 4):
        col = f"tag_{i}"
        if col in df.columns:
            counts = df[col].value_counts()
            for category, count in counts.items():
                if category:
                    category_counts[category] = category_counts.get(category, 0) + count
    
    if category_counts:
        stats["most_common_category"] = max(category_counts, key=category_counts.get)
        stats["categories_used"] = len(category_counts)
    
    return stats
