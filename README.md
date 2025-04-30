Overview
REformer is a computationally efficient architecture for extracting medication-related information from clinical texts. This model excels at identifying relationships between medications and their attributes (dosage, frequency, route, etc.) along with contextual information (start/stop dates, modifications, negations).
Key features:

Computationally efficient: 10-23x faster training compared to traditional methods
High performance: Comparable F1 scores to state-of-the-art methods
Multilingual support: Validated on both French and English clinical texts
Frame-based representation: Enhanced ability to represent complex medication regimens

Architecture
The architecture uses a transformer-based approach that classifies all relations simultaneously, avoiding the need to process each entity pair separately:

Input processing: Text tokenization and sliding window mechanism to handle long documents
Dual embedding: Combines token embeddings from a transformer model with entity class embeddings
Multi-head attention: Captures dependencies between tokens in the combined representation
Relation classification: For each token pair, predicts relation types using position-relative embeddings
Masked loss computation: Optimizes training by focusing only on actual entity pairs

