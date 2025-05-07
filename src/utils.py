"""
Utility functions for medical relation extraction.

This module provides helper functions for data processing, evaluation,
and visualization of medical relation extraction results.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from medkit.core.text import Entity, Relation, TextDocument


def entity_to_dict(entity):
    """
    Convert a medkit Entity to a dictionary representation.
    
    Args:
        entity: medkit Entity object
        
    Returns:
        Dictionary with entity information
    """
    return {
        "id": entity.uid,
        "label": entity.label,
        "text": entity.text,
        "spans": [(span.start, span.end) for span in entity.spans],
        "metadata": entity.metadata
    }


def relation_to_dict(relation, source_entity, target_entity):
    """
    Convert a medkit Relation to a dictionary representation.
    
    Args:
        relation: medkit Relation object
        source_entity: Source entity of the relation
        target_entity: Target entity of the relation
        
    Returns:
        Dictionary with relation information
    """
    return {
        "id": relation.uid,
        "label": relation.label,
        "source": {
            "id": source_entity.uid,
            "label": source_entity.label,
            "text": source_entity.text
        },
        "target": {
            "id": target_entity.uid,
            "label": target_entity.label,
            "text": target_entity.text
        }
    }


def doc_to_dict(doc):
    """
    Convert a medkit TextDocument to a dictionary representation.
    
    Args:
        doc: medkit TextDocument object
        
    Returns:
        Dictionary with document information
    """
    result = {
        "text": doc.text,
        "metadata": doc.metadata,
        "entities": [],
        "relations": []
    }
    
    # Add entities
    entities = doc.anns.get_entities()
    for entity in entities:
        result["entities"].append(entity_to_dict(entity))
    
    # Add relations
    relations = doc.anns.get_relations()
    for relation in relations:
        source_entity = doc.anns.get_by_id(relation.source_id)
        target_entity = doc.anns.get_by_id(relation.target_id)
        result["relations"].append(relation_to_dict(relation, source_entity, target_entity))
    
    return result


def compute_relation_metrics(true_docs, pred_docs):
    """
    Compute precision, recall, and F1 score for relation extraction.
    
    Args:
        true_docs: List of ground truth medkit TextDocument objects
        pred_docs: List of predicted medkit TextDocument objects
        
    Returns:
        Dictionary with precision, recall, and F1 metrics
    """
    # Extract true and predicted relations
    true_relations = []
    pred_relations = []
    
    # Create a mapping from document ID to document
    true_doc_map = {doc.metadata.get("path_to_text", ""): doc for doc in true_docs}
    pred_doc_map = {doc.metadata.get("path_to_text", ""): doc for doc in pred_docs}
    
    # Get common document IDs
    common_doc_ids = set(true_doc_map.keys()) & set(pred_doc_map.keys())
    
    for doc_id in common_doc_ids:
        true_doc = true_doc_map[doc_id]
        pred_doc = pred_doc_map[doc_id]
        
        # Get entities by brat_id
        true_entities = {
            ent.metadata.get("brat_id", ent.uid): ent 
            for ent in pred_doc.anns.get_entities()
        }
        
        # Extract relations as (source_id, target_id, label) tuples
        for relation in true_doc.anns.get_relations():
            source_id = relation.source_id
            target_id = relation.target_id
            source_entity = true_doc.anns.get_by_id(source_id)
            target_entity = true_doc.anns.get_by_id(target_id)
            
            # Use brat_id if available
            source_brat_id = source_entity.metadata.get("brat_id", source_id)
            target_brat_id = target_entity.metadata.get("brat_id", target_id)
            
            true_relations.append((source_brat_id, target_brat_id, relation.label))
        
        for relation in pred_doc.anns.get_relations():
            source_id = relation.source_id
            target_id = relation.target_id
            source_entity = pred_doc.anns.get_by_id(source_id)
            target_entity = pred_doc.anns.get_by_id(target_id)
            
            # Use brat_id if available
            source_brat_id = source_entity.metadata.get("brat_id", source_id)
            target_brat_id = target_entity.metadata.get("brat_id", target_id)
            
            pred_relations.append((source_brat_id, target_brat_id, relation.label))
    
    # Calculate metrics
    true_set = set(true_relations)
    pred_set = set(pred_relations)
    
    true_positives = len(true_set & pred_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-relation type metrics
    relation_types = set([rel[2] for rel in true_relations + pred_relations])
    per_type_metrics = {}
    
    for rel_type in relation_types:
        true_type = set([(s, t, l) for s, t, l in true_relations if l == rel_type])
        pred_type = set([(s, t, l) for s, t, l in pred_relations if l == rel_type])
        
        tp_type = len(true_type & pred_type)
        fp_type = len(pred_type - true_type)
        fn_type = len(true_type - pred_type)
        
        type_precision = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0
        type_recall = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
        
        per_type_metrics[rel_type] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "support": len(true_type)
        }
    
    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(true_relations)
        },
        "per_type": per_type_metrics
    }


def plot_relation_metrics(metrics, output_path=None):
    """
    Plot precision, recall, and F1 scores for each relation type.
    
    Args:
        metrics: Dictionary of metrics from compute_relation_metrics
        output_path: Path to save the plot (optional)
    """
    per_type = metrics["per_type"]
    relation_types = list(per_type.keys())
    
    precision = [per_type[rel]["precision"] for rel in relation_types]
    recall = [per_type[rel]["recall"] for rel in relation_types]
    f1 = [per_type[rel]["f1"] for rel in relation_types]
    support = [per_type[rel]["support"] for rel in relation_types]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1)[::-1]
    relation_types = [relation_types[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1 = [f1[i] for i in sorted_indices]
    support = [support[i] for i in sorted_indices]
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    x = np.arange(len(relation_types))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', color='#2196F3')
    ax1.bar(x, recall, width, label='Recall', color='#4CAF50')
    ax1.bar(x + width, f1, width, label='F1', color='#FF9800')
    
    # Configure primary y-axis
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('Relation Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(relation_types, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    
    # Create secondary y-axis for support
    ax2 = ax1.twinx()
    ax2.plot(x, support, 'r-', marker='o', label='Support')
    ax2.set_ylabel('Support (count)')
    ax2.legend(loc='upper right')
    
    # Add overall metrics as text
    overall = metrics["overall"]
    overall_text = (
        f"Overall Metrics:\n"
        f"Precision: {overall['precision']:.3f}\n"
        f"Recall: {overall['recall']:.3f}\n"
        f"F1: {overall['f1']:.3f}\n"
        f"Support: {overall['support']}"
    )
    plt.annotate(
        overall_text, 
        xy=(0.02, 0.97), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.title('Relation Extraction Performance by Relation Type')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    return fig


def visualize_doc_relations(doc, output_path=None):
    """
    Create a visualization of entities and relations in a document.
    
    Args:
        doc: medkit TextDocument object
        output_path: Path to save the visualization (optional)
    
    Returns:
        Matplotlib figure
    """
    # Get entities and relations
    entities = doc.anns.get_entities()
    relations = doc.anns.get_relations()
    
    # Create entity spans visualization
    text = doc.text
    entity_data = []
    
    for entity in entities:
        for span in entity.spans:
            entity_data.append({
                "start": span.start,
                "end": span.end,
                "label": entity.label,
                "text": text[span.start:span.end],
                "id": entity.uid
            })
    
    # Sort by start position
    entity_data = sorted(entity_data, key=lambda x: x["start"])
    
    # Create dataframe for visualization
    df = pd.DataFrame(entity_data)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot entities as spans
    y_pos = 0
    entity_positions = {}  # To track vertical positions of entities
    
    for _, row in df.iterrows():
        color = plt.cm.tab10(hash(row["label"]) % 10)
        ax1.barh(
            y_pos, 
            row["end"] - row["start"], 
            left=row["start"], 
            height=0.8, 
            color=color, 
            alpha=0.6
        )
        ax1.text(
            row["start"] + (row["end"] - row["start"]) / 2, 
            y_pos, 
            f"{row['text']} ({row['label']})", 
            ha='center', 
            va='center'
        )
        entity_positions[row["id"]] = y_pos
        y_pos += 1
    
    # Configure entity axis
    ax1.set_yticks([])
    ax1.set_xlabel('Character Position')
    ax1.set_title('Entity Spans in Document')
    
    # Plot relations as a network
    ax2.axis('off')
    ax2.set_title('Entity Relations')
    
    # Create a grid for entity positions
    cols = min(5, len(entities))
    rows = (len(entities) - 1) // cols + 1
    
    entity_grid_positions = {}
    for i, entity in enumerate(entities):
        row = i // cols
        col = i % cols
        entity_grid_positions[entity.uid] = (col, row)
    
    # Plot entities as nodes
    for entity in entities:
        x, y = entity_grid_positions[entity.uid]
        color = plt.cm.tab10(hash(entity.label) % 10)
        ax2.plot(x, -y, 'o', markersize=10, color=color)
        ax2.text(
            x, -y - 0.1, 
            f"{entity.text[:15]}... ({entity.label})", 
            ha='center', 
            va='top', 
            fontsize=8
        )
    
    # Plot relations as edges
    relation_data = []
    for relation in relations:
        source_entity = doc.anns.get_by_id(relation.source_id)
        target_entity = doc.anns.get_by_id(relation.target_id)
        
        source_pos = entity_grid_positions[relation.source_id]
        target_pos = entity_grid_positions[relation.target_id]
        
        ax2.plot(
            [source_pos[0], target_pos[0]], 
            [-source_pos[1], -target_pos[1]], 
            '-', 
            color='gray'
        )
        
        # Add midpoint label
        mid_x = (source_pos[0] + target_pos[0]) / 2
        mid_y = -(source_pos[1] + target_pos[1]) / 2
        ax2.text(
            mid_x, mid_y, 
            relation.label, 
            ha='center', 
            va='center', 
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            fontsize=8
        )
        
        relation_data.append({
            "source": source_entity.text,
            "source_label": source_entity.label,
            "target": target_entity.text,
            "target_label": target_entity.label,
            "relation": relation.label
        })
    
    # Add relation table
    if relation_data:
        rel_df = pd.DataFrame(relation_data)
        table_text = []
        for _, row in rel_df.iterrows():
            table_text.append(
                f"{row['source_label']}:[{row['source']}] -- "
                f"{row['relation']} --> "
                f"{row['target_label']}:[{row['target']}]"
            )
        
        ax2.text(
            0.5, -rows - 0.5, 
            "\n".join(table_text), 
            ha='center', 
            va='top',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
            fontsize=9
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    return fig

            for ent in true_doc.anns.get_entities()
        }
        pred_entities = {
            ent.metadata.get("brat_id", ent.uid): ent 