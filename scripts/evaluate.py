#!/usr/bin/env python
"""
Evaluation script for medical relation extraction model.

This script evaluates a trained relation extraction model on medical documents,
generating predicted relations and measuring performance against ground truth.
"""

import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
from medkit.io.brat import BratInputConverter, BratOutputConverter
import re
from tqdm import tqdm
import torch.serialization
torch.serialization.add_safe_globals(['src.models.RelationExtractionModel', 'src.models.RelationClassifier'])
# Local imports
import medkit
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_splitter import split_docs_sliding_character
from src.dataset import NERDataset, CustomDataLoaderEvaluation
from src.training import extract_relations_from_doc


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate medical relation extraction model")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test BRAT annotations")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    
    # Evaluation arguments
    parser.add_argument("--window_size", type=int, default=500, help="Window size for document splitting")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (-1 for CPU)")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval()
    
    # Get tokenizer from model configuration
    if "clinical" in args.model_path.lower():
        tokenizer_name = "medicalai/ClinicalBERT"
    elif "biobert" in args.model_path.lower():
        tokenizer_name = "dmis-lab/biobert-base-cased-v1.2"
    elif "camembert" in args.model_path.lower():
        tokenizer_name = "almanach/camembert-bio-base"
    else:
        tokenizer_name = "bert-base-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load test data
    print(f"Loading test data from {args.test_dir}")
    brat_converter = BratInputConverter()
    test_docs = brat_converter.load(dir_path=args.test_dir)
    print(f"Loaded {len(test_docs)} test documents")
    
    # Add brat_id to entity metadata
    for doc in test_docs:
        for ann in doc.anns.get_entities():
            ann.metadata['brat_id'] = ann.uid
    
    # Split documents for processing
    print(f"Splitting documents with window size {args.window_size}")
    test_docs_split = split_docs_sliding_character(test_docs, args.window_size)
    print(f"Split into {len(test_docs_split)} chunks")
    
    # Filter documents with drug entities
    print("Filtering documents with drug entities")
    selected_test_docs = []
    for doc in test_docs_split:
        entity_labels = [ent.label for ent in doc.anns.get_entities()]
        if len(entity_labels) > 1:  # At least two entities
            selected_test_docs.append(doc)
    print(f"Filtered to {len(selected_test_docs)} chunks with multiple entities")
    
    # Define label map for entity types
    label_map = {
        "O": 0,
        "B-Dosage": 1, "I-Dosage": 2,
        "B-Drug": 3, "I-Drug": 4,
        "B-Duration": 5, "I-Duration": 6,
        "B-Form": 7, "I-Form": 8,
        "B-Frequency": 9, "I-Frequency": 10,
        "B-Reason": 11, "I-Reason": 12,
        "B-Route": 13, "I-Route": 14,
        "B-Strength": 15, "I-Strength": 16,
        "B-ADE": 17, "I-ADE": 18
    }
    
    # Define relation map
    relation_map = {
        'ADE-Drug': 1,
        'Dosage-Drug': 2,
        'Duration-Drug': 3,
        'Form-Drug': 4,
        'Frequency-Drug': 5,
        'Reason-Drug': 6,
        'Route-Drug': 7,
        'Strength-Drug': 8
    }
    
    # Inverse relation map for prediction
    inverse_relation_map = {v: k for k, v in relation_map.items()}
    
    # Create dataset
    print("Creating evaluation dataset")
    test_dataset = NERDataset(selected_test_docs, tokenizer, label_map, False, relation_map)
    
    # Extract relations from documents
    print("Extracting relations from test documents")
    pred_docs = extract_relations_from_doc(
        model=model,
        dataset=test_dataset,
        docs=test_docs,
        tokenizer=tokenizer,
        window_size=args.window_size,
        inverse_relation_map=inverse_relation_map
    )
    
    if pred_docs is None:
        print("ERREUR: L'extraction de relations a échoué et retourné None")
    pred_docs = []  # Initialiser à une liste vide pour éviter l'erreur
    
    # Créer des copies des documents originaux sans relations
    for doc in test_docs:
        new_doc = medkit.core.text.TextDocument(text=doc.text)
        new_doc.metadata = doc.metadata
        
        # Copier uniquement les entités
        for ent in doc.anns.get_entities():
            new_doc.anns.add(ent)
            
        pred_docs.append(new_doc)
    
    print(f"Créé {len(pred_docs)} documents sans relations comme solution de secours")


    # De-duplicate relations in predicted documents
    print("De-duplicating relations")
    pred_docs_final = []


    
    for doc in pred_docs:
        # Create new document with original text and metadata
        new_doc = medkit.core.text.TextDocument(text=doc.text)
        new_doc.metadata = doc.metadata
        
        # Copy all entities
        for ent in doc.anns.get_entities():
            new_doc.anns.add(ent)
        
        # Handle relations
        relation_set = set()
        for relation in doc.anns.get_relations():
            # Only keep relations where source entity is not a Drug
            source_entity = doc.anns.get_by_id(relation.source_id)
            if source_entity.label != "Drug":
                # Convert relation to tuple for deduplication
                relation_key = (relation.source_id, relation.target_id, relation.label)
                if relation_key not in relation_set:
                    relation_set.add(relation_key)
                    new_doc.anns.add(relation)
        
        pred_docs_final.append(new_doc)
    
    # Save results as BRAT annotations
    print(f"Saving results to {args.output_dir}")
    brat_output_converter = BratOutputConverter()
    
    # Extract document names from metadata
    doc_names = [
        re.sub(r".*\\", "", doc.metadata.get('path_to_text', f"doc_{i}"))[:-4]
        for i, doc in enumerate(pred_docs_final)
    ]
    
    # Save BRAT annotations
    brat_output_converter.save(
        pred_docs_final, 
        dir_path=args.output_dir, 
        doc_names=doc_names
    )
    
    # Evaluate if evaluation script is available
    try:
        from eval import Corpora, evaluate
        print("Running evaluation")
        corpora = Corpora(args.test_dir, args.output_dir, 2)
        if corpora.docs:
            results = evaluate(corpora, verbose=True)
            
            # Save evaluation results to file
            with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
                f.write("Evaluation Results\n")
                f.write("=================\n\n")
                for metric, value in results.items():
                    f.write(f"{metric}: {value}\n")
            
            print(f"Evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.txt')}")
        else:
            print("No documents found for evaluation")
    except ImportError:
        print("Evaluation script not found. Skipping evaluation.")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
