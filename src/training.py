"""
Training utilities for medical relation extraction models.

This module provides functions for training and evaluating relation extraction models,
including cross-validation, metrics calculation, and model validation.
"""

import torch
import torch.nn as nn
import torch.cuda
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import gc
import datetime
from medkit.core.text import TextDocument, Relation


def train_model(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    loss_fn, 
    num_epochs, 
    accelerator=None, 
    val_dataloader=None, 
    accumulation_steps=1
):
    """
    Train a relation extraction model.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        num_epochs: Number of training epochs
        accelerator: Accelerator for distributed training (optional)
        val_dataloader: Validation data loader (optional)
        accumulation_steps: Gradient accumulation steps (default=1)
        
    Returns:
        Trained model and training history
    """
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    start_time = datetime.datetime.now()
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        z = 0
        model.train()
        
        # Training loop
        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for batch in t:
                z += 1
                input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, _ = batch
                
                # Forward pass
                relation_matrix = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
                
                # Calculate loss
                relation_matrix = relation_matrix.float()
                target_relations = target_relations.long()
                relation_loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])
                
                loss = relation_loss
                total_loss += loss.item()
                
                # Scale loss if using gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                if accelerator:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                
                # Update weights after accumulation steps
                if (z + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                t.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation step
        if val_dataloader:
            val_loss, _, _ = validate_model(model, val_dataloader, loss_fn)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
    
    end_time = datetime.datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f'Training completed in {training_time:.2f} seconds')
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 'training_time': training_time}


def validate_model(model, dataloader, loss_fn):
    """
    Validate a relation extraction model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        loss_fn: Loss function
        
    Returns:
        Average loss, F1 score, and model predictions
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_samples = 0
    predictions = []
    all_preds = []
    all_targets = []

    with torch.no_grad():  # Disable gradient computation
        with tqdm(dataloader, desc='Validation', unit='batch') as t:
            for batch in t:
                input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, _ = batch
                
                # Forward pass
                relation_matrix = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
                predictions.append(relation_matrix)
                
                # Calculate loss
                relation_matrix = relation_matrix.float()
                target_relations = target_relations.long()
                loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])
                
                # Accumulate loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
                
                # Get predicted classes
                preds = torch.argmax(relation_matrix[pair_mask == 1], dim=-1)
                targets = target_relations[pair_mask == 1]
                
                # Accumulate predictions and targets for metrics
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                t.set_postfix({'val_loss': loss.item()})

    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    # Calculate F1 score (excluding 'O' class)
    f1 = calculate_f1_score(all_targets, all_preds, average='micro', exclude_classes=[0])

    return avg_loss, f1, predictions


def calculate_f1_score(y_true, y_pred, average='macro', exclude_classes=[0]):
    """
    Calculate F1 score, precision, and recall for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Type of averaging ('macro', 'micro', 'weighted')
        exclude_classes: Classes to exclude from metrics (e.g., [0] to exclude 'O' class)
        
    Returns:
        F1 score
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create a mask to filter out excluded classes
    mask = np.isin(y_true, exclude_classes, invert=True)
    
    # Apply the mask
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate metrics
    if len(y_true_filtered) == 0:
        return 0.0  # Return 0 if no valid labels remain
        
    f1 = f1_score(y_true_filtered, y_pred_filtered, average=average)
    
    return f1


def train_with_cross_validation(
    dataset, 
    model_class, 
    tokenizer,
    embedding_dim=16, 
    num_heads=8, 
    hidden_dim=128, 
    pos_dim=256, 
    num_folds=5, 
    num_epochs=10, 
    batch_size=8, 
    lr=1e-4, 
    max_length=512,
    pretrained_model_name="medicalai/ClinicalBERT",
    num_entity_types=21,
    num_classes_relation=10,
    log_dir='./logs'
):
    """
    Train a model with cross-validation.
    
    Args:
        dataset: Dataset for training
        model_class: Model class to instantiate
        tokenizer: Tokenizer to use
        embedding_dim: Dimension of entity embeddings
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension size
        pos_dim: Position embedding dimension
        num_folds: Number of cross-validation folds
        num_epochs: Number of training epochs per fold
        batch_size: Batch size
        lr: Learning rate
        max_length: Maximum sequence length
        pretrained_model_name: Name of pretrained model
        num_entity_types: Number of entity types
        num_classes_relation: Number of relation classes
        log_dir: Directory for TensorBoard logs
        
    Returns:
        Cross-validation results
    """
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Check compatibility of embedding dimension and number of heads
    from transformers import AutoModel
    temp_model = AutoModel.from_pretrained(pretrained_model_name)
    bert_hidden_size = temp_model.config.hidden_size
    
    if (bert_hidden_size + embedding_dim) % num_heads != 0:
        raise ValueError(f"Hidden size + embedding_dim ({bert_hidden_size + embedding_dim}) must be divisible by num_heads ({num_heads})")
    
    # Initialize lists for results
    all_train_losses = []
    all_val_losses = []
    all_val_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Create model for this fold
        from transformers import AutoModel
        bert_model = AutoModel.from_pretrained(pretrained_model_name)
        bert_model.resize_token_embeddings(len(tokenizer))
        
        # Initialize relation classifier
        from src.models import RelationClassifier
        relation_classifier = RelationClassifier(
            input_dim=bert_model.config.hidden_size,
            num_entity_types=num_entity_types,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            pos_dim=pos_dim,
            num_classes_relation=num_classes_relation
        )
        
        # Create model
        model = model_class(bert_model, relation_classifier)
        
        # Create data loaders for this fold
        from src.dataset import CustomDataLoader
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        train_dataloader = CustomDataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            max_length=max_length
        )
        val_dataloader = CustomDataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            max_length=max_length
        )
        
        # Setup training
        loss_fn = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=lr)
        num_training_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        
        # Setup accelerator for distributed training
        accelerator = Accelerator()
        model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader = accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, loss_fn, val_dataloader
        )
        
        # Initialize lists for this fold
        train_losses = []
        val_losses = []
        val_f1_scores = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, _ = batch
                
                # Forward pass
                relation_matrix = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
                
                # Calculate loss
                relation_matrix = relation_matrix.float()
                target_relations = target_relations.long()
                relation_loss = loss_fn(relation_matrix[pair_mask == 1], target_relations[pair_mask == 1])
                
                loss = relation_loss
                total_loss += loss.item()
                
                # Backward pass and update
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Calculate average loss
            avg_train_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss, val_f1, _ = validate_model(model, val_dataloader, loss_fn)
            val_losses.append(val_loss)
            val_f1_scores.append(val_f1)
            
            # Log to TensorBoard
            writer.add_scalar(f'Loss/Train/Fold{fold+1}', avg_train_loss, epoch+1)
            writer.add_scalar(f'Loss/Val/Fold{fold+1}', val_loss, epoch+1)
            writer.add_scalar(f'F1/Val/Fold{fold+1}', val_f1, epoch+1)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
            
            # Early stopping check
            if (epoch == 3 and val_loss > 1) or (epoch == 6 and val_loss > 0.8) or (epoch == 1 and avg_train_loss > 1.7):
                print("Early stopping triggered")
                break
        
        # Save results for this fold
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_val_f1_scores.append(val_f1_scores)
        
        # Clean up GPU memory
        del model, optimizer, scheduler, train_dataloader, val_dataloader, loss_fn
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate average performance across folds
    # Pad lists to same length (some folds might have stopped early)
    def pad_with_nan(array_list):
        max_length = max(len(arr) for arr in array_list)
        padded_list = []
        for arr in array_list:
            padding_length = max_length - len(arr)
            padded_arr = np.pad(arr, (0, padding_length), constant_values=np.nan)
            padded_list.append(padded_arr)
        return np.array(padded_list)
    
    all_train_losses_padded = pad_with_nan(all_train_losses)
    all_val_losses_padded = pad_with_nan(all_val_losses)
    all_val_f1_scores_padded = pad_with_nan(all_val_f1_scores)
    
    # Calculate means, ignoring NaN values
    avg_train_loss = np.nanmean(all_train_losses_padded, axis=0)
    avg_val_loss = np.nanmean(all_val_losses_padded, axis=0)
    avg_val_f1 = np.nanmean(all_val_f1_scores_padded, axis=0)
    
    # Log average performance
    for epoch in range(len(avg_train_loss)):
        writer.add_scalar('Loss/Train/Avg', avg_train_loss[epoch], epoch+1)
        writer.add_scalar('Loss/Val/Avg', avg_val_loss[epoch], epoch+1)
        writer.add_scalar('F1/Val/Avg', avg_val_f1[epoch], epoch+1)
    
    # Print average performance
    print("\nAverage performance across folds:")
    for epoch in range(len(avg_train_loss)):
        print(f'Epoch {epoch+1}/{num_epochs}, Avg Train Loss: {avg_train_loss[epoch]:.4f}, '
              f'Avg Val Loss: {avg_val_loss[epoch]:.4f}, Avg Val F1: {avg_val_f1[epoch]:.4f}')
    
    writer.close()
    
    return {
        'train_losses': all_train_losses, 
        'val_losses': all_val_losses, 
        'val_f1_scores': all_val_f1_scores,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_val_f1': avg_val_f1
    }


def align_predictions_with_entities(preds, batch, inverse_relation_map):
    """
    Align model predictions with entity IDs.
    
    Args:
        preds: Model predictions
        batch: Batch data
        inverse_relation_map: Mapping from relation IDs to labels
        
    Returns:
        List of (entity1_id, entity2_id, relation_type) tuples
    """
    input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, ids_token = batch
    
    result = []  # Ground truth relations
    predicted = []  # Predicted relations
    
    for z in range(len(input_ids)):
        # Extract entity IDs from the batch
        ids = np.array([x for y in ids_token[z] for x in y])
        
        ids_len = len(input_ids[z])
        ids = [x for i, x in enumerate(ids) if i <= ids_len]
        ids = ids + [None] * (ids_len - len(ids))
        
        # Create boolean masks for valid indices
        valid_mask = np.array([x is not None for x in ids])
        
        # Get all pairs of indices
        i, j = np.meshgrid(np.arange(len(ids)), np.arange(len(ids)), indexing='ij')
        
        # Apply masks to get valid pairs
        valid_pairs = valid_mask[i] & valid_mask[j]
        
        # Convert tensors to numpy arrays
        target_relations_np = target_relations[z].cpu().numpy() if isinstance(target_relations[z], torch.Tensor) else target_relations[z]
        pred_np = preds[z].cpu().numpy() if isinstance(preds[z], torch.Tensor) else preds[z]
        
        # Find non-zero relation pairs
        target_matches = np.where((target_relations_np != 0) & valid_mask[:, None] & valid_mask[None, :])
        pred_matches = np.where((pred_np != 0) & valid_mask[:, None] & valid_mask[None, :])
        
        # Add ground truth relations to result
        result.extend([
            (ids[i], ids[j], inverse_relation_map.get(target_relations_np[i, j], f"Unknown-{target_relations_np[i, j]}"))
            for i, j in zip(target_matches[0], target_matches[1])
        ])
        
        # Add predicted relations to predicted
        predicted.extend([
            (ids[i], ids[j], inverse_relation_map.get(pred_np[i, j], f"Unknown-{pred_np[i, j]}"))
            for i, j in zip(pred_matches[0], pred_matches[1])
        ])
    
    return result, predicted


    return result, predicted


def extract_relations_from_doc(model, dataset, docs, tokenizer, window_size, inverse_relation_map):
    """
    Extract relations from documents using a trained model.
    
    Args:
        model: Trained relation extraction model
        dataset: Dataset containing the document chunks
        docs: Original documents
        tokenizer: Tokenizer
        window_size: Window size used for splitting
        inverse_relation_map: Mapping from relation IDs to labels
        
    Returns:
        List of documents with extracted relations
    """
    from src.dataset import CustomDataLoaderEvaluation
    from tqdm import tqdm
    import gc
    
    # Initialize accelerator for inference
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    
    # List to store documents with predicted relations
    pred_docs = []
    
    # Process each document
    for doc in tqdm(docs, desc="Extracting relations"):
        # Create a new document with the original text
        new_doc = TextDocument(text=doc.text)
        new_doc.metadata = doc.metadata
        
        # Find chunks that belong to this document
        interest_chunks = [
            i for i, x in enumerate(dataset.docs) 
            if x.metadata.get("path_to_text") == doc.metadata.get("path_to_text")
        ]
        
        if len(interest_chunks) > 0:
            # Get dataset items for chunks of this document
            dataset_interest = [dataset[i] for i in interest_chunks]
            chunks_interested = [dataset.docs[i] for i in interest_chunks]
            
            # Create data loader for evaluation
            val_dataloader = CustomDataLoaderEvaluation(
                dataset_interest, 
                batch_size=1, 
                shuffle=False, 
                max_length=512
            )
            val_dataloader = accelerator.prepare(val_dataloader)
            
            # Extract relations
            relations = []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, labels, target_relations, pair_mask, sentence_mask, _ = batch
                    
                    # Get model predictions
                    relation_logits = model(input_ids, attention_mask, labels, pair_mask, sentence_mask)
                    preds = torch.argmax(relation_logits, -1)
                    
                    # Align predictions with entity IDs
                    _, relation_predictions = align_predictions_with_entities(preds, batch, inverse_relation_map)
                    
                    # Add predictions to relations list
                    relations = list(set(relations + relation_predictions))
            
            # Group relations by entity pair
            relations_dict = {}
            for entity1, entity2, label in relations:
                if (entity1, entity2) not in relations_dict:
                    relations_dict[(entity1, entity2)] = []
                relations_dict[(entity1, entity2)].append(label)
            
            # Copy entities from original document
            for ent in doc.anns.get_entities():
                new_doc.anns.add(ent)
            
            # Create relations between entities
            for chunk_idx in range(len(chunks_interested)):
                # Get drug entities in this chunk
                drug_entity_ids = [
                    ent.uid for ent in chunks_interested[chunk_idx].anns.get_entities() 
                    if ent.label == 'Drug'
                ]
                
                # Get entity IDs that are targets of relations
                target_ids = [x[1] for x in relations_dict.keys()]
                
                # Find drug entities that are targets of relations
                drug_ids_with_relations = [drug_id for drug_id in drug_entity_ids if drug_id in target_ids]
                
                # Add relations for each drug entity
                for drug_id in drug_ids_with_relations:
                    # Find related entities
                    related_entity_ids = [
                        entity_id for (entity_id, drug_entity_id) in relations_dict.keys() 
                        if drug_entity_id == drug_id
                    ]
                    
                    for entity_id in related_entity_ids:
                        # Get entity label
                        entity_label = chunks_interested[chunk_idx].anns.get_by_id(entity_id).label
                        
                        # Skip if both entities are drugs
                        if entity_label == "Drug":
                            continue
                        
                        # Add relation for each label
                        for relation_label in set(relations_dict[(entity_id, drug_id)]):
                            # Map chunk entity IDs to document entity IDs
                            chunk_source_id = entity_id
                            chunk_target_id = drug_id
                            
                            # Get original BRAT IDs
                            brat_source_id = chunks_interested[chunk_idx].anns.get_by_id(chunk_source_id).metadata.get("brat_id")
                            brat_target_id = chunks_interested[chunk_idx].anns.get_by_id(chunk_target_id).metadata.get("brat_id")
                            
                            # Find corresponding entities in the original document
                            doc_source_id = next(
                                (e.uid for e in doc.anns.get_entities() if e.metadata.get("brat_id") == brat_source_id), 
                                None
                            )
                            doc_target_id = next(
                                (e.uid for e in doc.anns.get_entities() if e.metadata.get("brat_id") == brat_target_id), 
                                None
                            )
                            
                            if doc_source_id and doc_target_id:
                                # Create relation
                                new_doc.anns.add(Relation(
                                    label=relation_label,
                                    source_id=doc_source_id,
                                    target_id=doc_target_id
                                ))
            
            # Clean up GPU memory
            del val_dataloader, batch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Add document to results
        pred_docs.append(new_doc)
    
    return pred_docs
