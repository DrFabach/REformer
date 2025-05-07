"""
Medical Relation Extraction package initialization.

This package provides tools for extracting relations between medical entities
in clinical text documents using BERT-based models.
"""

from .document_splitter import DocumentSplitterSlidingCharacter, split_docs_sliding_character
from .dataset import NERDataset, CustomDataLoader, CustomDataLoaderEvaluation
from .models import RelationClassifier, RelationExtractionModel
from .training import train_model, validate_model, extract_relations_from_doc
from .utils import compute_relation_metrics, plot_relation_metrics, visualize_doc_relations

__version__ = "0.1.0"
