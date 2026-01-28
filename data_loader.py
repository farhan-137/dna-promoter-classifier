"""
Data loader for E. Coli Promoter sequences
UCI ML Repository dataset - 106 sequences (57 nucleotides each)

@author: parzival
"""

import random


def load_promoter_data(filepath):
    """
    Parse the UCI promoter data file.
    Format: label,gene_name,\t\tsequence
    
    Returns:
        sequences: list of DNA strings
        labels: list of +1 (promoter) or -1 (non-promoter)
        names: list of gene names
    """
    sequences = []
    labels = []
    names = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # split by comma
            parts = line.split(',')
            if len(parts) < 3:
                continue
            
            label_str = parts[0].strip()
            gene_name = parts[1].strip()
            # the sequence might have tabs before it
            seq = parts[2].replace('\t', '').strip().lower()
            
            # convert label
            label = 1 if label_str == '+' else -1
            
            sequences.append(seq)
            labels.append(label)
            names.append(gene_name)
    
    return sequences, labels, names


def train_test_split(sequences, labels, test_ratio=0.2, seed=42):
    """
    Split data into training and test sets.
    Tries to keep class balance roughly equal in both sets.
    """
    random.seed(seed)
    
    # separate by class
    pos_indices = [i for i, y in enumerate(labels) if y == 1]
    neg_indices = [i for i, y in enumerate(labels) if y == -1]
    
    # shuffle each class
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)
    
    # split each class
    n_pos_test = max(1, int(len(pos_indices) * test_ratio))
    n_neg_test = max(1, int(len(neg_indices) * test_ratio))
    
    test_indices = pos_indices[:n_pos_test] + neg_indices[:n_neg_test]
    train_indices = pos_indices[n_pos_test:] + neg_indices[n_neg_test:]
    
    # shuffle the final sets
    random.shuffle(test_indices)
    random.shuffle(train_indices)
    
    X_train = [sequences[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_test = [sequences[i] for i in test_indices]
    y_test = [labels[i] for i in test_indices]
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # quick test
    import os
    
    data_path = os.path.join(os.path.dirname(__file__), "data", "promoters.data")
    seqs, labels, names = load_promoter_data(data_path)
    
    print(f"Loaded {len(seqs)} sequences")
    print(f"Promoters: {sum(1 for y in labels if y == 1)}")
    print(f"Non-promoters: {sum(1 for y in labels if y == -1)}")
    print(f"\nExample sequence ({names[0]}): {seqs[0][:30]}...")
    print(f"Sequence length: {len(seqs[0])}")
    
    X_train, y_train, X_test, y_test = train_test_split(seqs, labels)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
