"""
Models for medical relation extraction.

This module implements neural network models for extracting relations
between medical entities in clinical text.
"""

import torch
import torch.nn as nn
import math


class RelativePositionEmbedding(nn.Module):
    """
    Learnable embedding for relative positions between tokens.
    
    This helps the model understand the distance between entities
    in the sequence, which is important for relation extraction.
    """
    def __init__(self, max_seq_len, d_model):
        super(RelativePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rel_pos_embedding = nn.Parameter(torch.Tensor(2 * max_seq_len - 1, d_model))
        nn.init.xavier_uniform_(self.rel_pos_embedding)
    
    def forward(self, batch_size, seq_len):
        """
        Create relative position embeddings for each token pair.
        
        Args:
            batch_size: Number of examples in the batch
            seq_len: Sequence length
            
        Returns:
            Tensor of shape (batch_size, seq_len, seq_len, d_model)
        """
        row_idx = torch.arange(seq_len, device=self.rel_pos_embedding.device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=self.rel_pos_embedding.device).unsqueeze(0)
        relative_positions = row_idx - col_idx + self.max_seq_len - 1
        relative_positions = relative_positions.unsqueeze(0).expand(batch_size, -1, -1)
        return self.rel_pos_embedding[relative_positions]


class CustomLayer(nn.Module):
    """
    Custom feed-forward network for relation classification.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CustomLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class PairwiseLayer(nn.Module):
    """
    Layer for classifying relations between token pairs.
    """
    def __init__(self, d, num_classes, hidden_dim=256):
        super(PairwiseLayer, self).__init__()
        self.custom_layer = CustomLayer(d * 2, hidden_dim, num_classes)  

    def forward(self, x, mask, position):
        """
        Forward pass for classifying token pairs.
        
        Args:
            x: Token representations (batch_size, seq_len, d)
            mask: Boolean mask for valid token pairs (batch_size, seq_len, seq_len)
            position: Relative position embeddings (batch_size, seq_len, seq_len, pos_dim)
            
        Returns:
            Classification scores for each token pair (batch_size, seq_len, seq_len, num_classes)
        """
        batch_size, seq_len, d = x.shape

        # Initialize output tensor
        output = torch.zeros(
            batch_size, seq_len, seq_len, self.custom_layer.fc[-1].out_features,
            device=x.device
        )

        # Get indices of token pairs to classify
        mask_indices = mask.nonzero(as_tuple=True)
                
        # Extract position embeddings for these pairs
        positionx = position[mask_indices[0], mask_indices[1], mask_indices[2]]
        
        # Extract token representations for these pairs
        x_i = x[mask_indices[0], mask_indices[1]]
        x_j = x[mask_indices[0], mask_indices[2]]

        # Concatenate token features and position embedding
        combined_features = torch.cat((x_i, x_j, positionx), dim=-1)

        # Apply classification layer
        output_values = self.custom_layer(combined_features)

        # Assign classification scores to output tensor
        output[mask_indices[0], mask_indices[1], mask_indices[2]] = output_values

        return output


class RelationClassifier(nn.Module):
    """
    Main relation classifier module using multi-head attention.
    """
    def __init__(
        self, 
        input_dim, 
        num_entity_types, 
        embedding_dim, 
        num_heads=4, 
        hidden_dim=256, 
        max_seq_len=512, 
        pos_dim=10, 
        num_classes_relation=5
    ):
        super(RelationClassifier, self).__init__()
        
        # Transform layer for combining token embeddings with entity type embeddings
        self.transform = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, input_dim + embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(input_dim + embedding_dim)
        )
        
        self.num_heads = num_heads
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity_types, embedding_dim=embedding_dim)
        self.attention = nn.MultiheadAttention(input_dim + embedding_dim, num_heads, dropout=0.1)
        self.layer_norm = nn.LayerNorm(input_dim + embedding_dim)
        
        # Pairwise classification layer
        self.pairwise_layer = PairwiseLayer(
            input_dim + embedding_dim + pos_dim, 
            hidden_dim=hidden_dim, 
            num_classes=num_classes_relation
        )
        
        # Relative position embeddings
        self.rel_pos_embedding = RelativePositionEmbedding(max_seq_len, pos_dim*2)

    def forward(self, x, entity_labels, attention_mask, sentence_mask):
        """
        Forward pass for relation classification.
        
        Args:
            x: Token representations from BERT (batch_size, seq_len, hidden_dim)
            entity_labels: Entity type labels for each token (batch_size, seq_len)
            attention_mask: Mask for valid token pairs (batch_size, seq_len, seq_len)
            sentence_mask: Mask for attending to tokens in the same sentence (batch_size, seq_len, seq_len)
            
        Returns:
            Classification scores for each token pair (batch_size, seq_len, seq_len, num_classes)
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Get entity type embeddings
        entity_labels = entity_labels.long()
        entity_embeddings = self.entity_embedding(entity_labels)

        # Combine token embeddings with entity type embeddings
        combined = torch.cat([x, entity_embeddings], dim=-1)
        transformed = self.transform(combined)
        
        # Apply sentence mask to attention if provided
        if sentence_mask is not None:
            # Expand mask for multi-head attention
            sentence_mask = sentence_mask.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
            sentence_mask = sentence_mask.reshape(batch_size * self.num_heads, seq_len, seq_len)
            # Convert to boolean mask (True for tokens that can attend to each other)
            sentence_mask = sentence_mask.float().masked_fill(sentence_mask == 0, False).masked_fill(sentence_mask == 1, True)
       
        # Reshape for attention (seq_len, batch_size, hidden_dim)
        transformed = transformed.permute(1, 0, 2)
        
        # Apply self-attention
        attn_output, _ = self.attention(transformed, transformed, transformed, attn_mask=sentence_mask)
        
        # Reshape back to (batch_size, seq_len, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        # Apply layer normalization
        output = self.layer_norm(attn_output)

        # Convert attention mask to boolean
        attention_mask = attention_mask.reshape(batch_size, seq_len, seq_len).float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, False).masked_fill(attention_mask == 1, True).bool()
        
        # Get relative position embeddings
        rel_pos_emb = self.rel_pos_embedding(batch_size, seq_len)
        
        # Apply pairwise classification
        output = self.pairwise_layer(output, attention_mask, rel_pos_emb)
        
        return output


class RelationExtractionModel(nn.Module):
    """
    Complete model combining BERT with relation classifier.
    """
    def __init__(self, bert_model, relation_classifier):
        super(RelationExtractionModel, self).__init__()
        self.bert = bert_model
        self.relation_classifier = relation_classifier

    def forward(self, input_ids, attention_mask, entity_labels, pair_mask, sentence_mask):
        """
        Forward pass for the complete model.
        
        Args:
            input_ids: Token IDs from BERT tokenizer (batch_size, seq_len)
            attention_mask: Attention mask for BERT (batch_size, seq_len)
            entity_labels: Entity type labels for each token (batch_size, seq_len)
            pair_mask: Mask for valid token pairs (batch_size, seq_len, seq_len)
            sentence_mask: Mask for attending to tokens in the same sentence (batch_size, seq_len, seq_len)
            
        Returns:
            Classification scores for each token pair (batch_size, seq_len, seq_len, num_classes)
        """
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state

        # Apply relation classifier
        relation_matrix = self.relation_classifier(
            token_embeddings, 
            entity_labels, 
            pair_mask, 
            sentence_mask
        )

        return relation_matrix
