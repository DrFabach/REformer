#!/usr/bin/env python
"""
Training script for medical relation extraction model.

This script trains a relation extraction model on medical documents,
using a BERT-based architecture to identify relations between medical entities.
"""

import os
import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel
from medkit.io.brat import BratInputConverter

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_splitter import split_docs_sliding_character
from src.dataset import NERDataset
from src.models import RelationClassifier, RelationExtractionModel
from src.training import train_model, validate_model


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train medical relation extraction model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing BRAT annotations")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save the model")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="clinical_bert", 
                      choices=["clinical_bert", "biobert", "bert", "camembert"], 
                      help="Type of BERT model to use")
    parser.add_argument("--window_size", type=int, default=500, help="Window size for document splitting")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Dimension of entity embeddings")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--pos_dim", type=int, default=256, help="Position embedding dimension")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use (-1 for CPU)")
    
    return parser.parse_args()


def get_pretrained_model_name(model_type):
    """Get the HuggingFace model name based on model type."""
    model_name_map = {
        "clinical_bert": "medicalai/ClinicalBERT",
        "biobert": "dmis-lab/biobert-base-cased-v1.2",
        "bert": "bert-base-uncased",
        "camembert": "almanach/camembert-bio-base"
    }
    return model_name_map.get(model_type, "medicalai/ClinicalBERT")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get pretrained model name
    pretrained_model_name = get_pretrained_model_name(args.model_type)
    print(f"Using pretrained model: {pretrained_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    # Load BRAT annotations
    print(f"Loading data from {args.data_dir}")
    brat_converter = BratInputConverter()
    docs = brat_converter.load(dir_path=args.data_dir)
    print(f"Loaded {len(docs)} documents")
    
    # Split documents into sliding windows
    print(f"Splitting documents with window size {args.window_size}")
    split_docs = split_docs_sliding_character(docs, args.window_size)
    print(f"Split into {len(split_docs)} chunks")
    
    # Filter documents with at least one drug entity
    print("Filtering documents with drug entities")
    selected_docs = []
    for doc in split_docs:
        entity_labels = [ent.label for ent in doc.anns.get_entities()]
        if "Drug" in entity_labels and len(entity_labels) > 1:
            selected_docs.append(doc)
    print(f"Filtered to {len(selected_docs)} chunks with drug entities")
    
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
    
    # Create dataset
    print("Creating dataset")
    dataset = NERDataset(selected_docs, tokenizer, label_map, False, relation_map)
    
    # Calculate train/val split indices
    dataset_size = len(dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Create data loaders
    from src.dataset import CustomDataLoader
    train_dataloader = CustomDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        max_length=args.max_length
    )
    val_dataloader = CustomDataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        max_length=args.max_length
    )
    
    # Create model
    print("Creating model")
    bert_model = AutoModel.from_pretrained(pretrained_model_name)
    bert_model.resize_token_embeddings(len(tokenizer))
    
    # Initialize relation classifier
    relation_classifier = RelationClassifier(
        input_dim=bert_model.config.hidden_size,
        num_entity_types=len(label_map),
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        pos_dim=args.pos_dim,
        num_classes_relation=len(relation_map) + 1  # +1 for "no relation" class
    )
    
    # Create model
    model = RelationExtractionModel(bert_model, relation_classifier)
    model.to(device)
    
    # Setup training
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_dataloader) * args.num_epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    # Use accelerator for distributed training
    from accelerate import Accelerator
    accelerator = Accelerator()
    model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader
    )
    
    # Train model
    print(f"Starting training for {args.num_epochs} epochs")
    model, history = train_model(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_epochs=args.num_epochs,
        accelerator=accelerator,
        val_dataloader=val_dataloader
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_type}_{args.window_size}")
    print(f"Saving model to {model_path}")
    torch.save(model, model_path)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
