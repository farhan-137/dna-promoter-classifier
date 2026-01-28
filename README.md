# DNA Promoter Sequence Classifier

A Support Vector Machine classifier for E. Coli DNA promoter sequences, built from scratch using String Kernels and the SMO optimization algorithm.

## What This Does

Classifies DNA sequences as **promoters** (regions that initiate gene expression) or **non-promoters** using machine learning.

The key insight: instead of hand-crafting features, we use the **kernel trick** to implicitly work in a high-dimensional feature space (k-mer frequencies) without ever building those massive feature vectors.

## The Math Behind It

For DNA sequences with a 4-letter alphabet {A, C, G, T}:
- k=3 mer features: 4³ = 64 dimensions
- k=5 mer features: 4⁵ = 1,024 dimensions  
- k=10 mer features: 4¹⁰ = 1,048,576 dimensions!

For proteins with 20 amino acids:
- k=5 mer features: 20⁵ = 3,200,000 dimensions

The **kernel trick** lets us compute the dot product in this space without actually building the vectors. We just need K(s1, s2) = Σ count(kmer in s1) × count(kmer in s2).

## Files

```
dna_classifier/
├── data/
│   └── promoters.data      # UCI dataset (106 sequences)
├── data_loader.py          # Parse and split data
├── string_kernel.py        # Spectrum, mismatch, subsequence kernels
├── svm_scratch.py          # SVM with SMO algorithm
├── train.py                # Main training script
└── README.md
```

## Usage

```bash
cd dna_classifier
python train.py
```

## Kernels Implemented

1. **Spectrum Kernel** - Counts exact k-mer matches
2. **Normalized Spectrum** - Spectrum normalized by self-similarity
3. **Mismatch Kernel** - Allows up to m mismatches per k-mer (more robust)
4. **Subsequence Kernel** - Counts non-contiguous subsequences with gap penalty

## Optimizations

- **Pre-computed Gram Matrix**: All pairwise kernel values computed once before training. This cuts down on redundant string comparisons during SMO.

## Dataset

E. Coli Promoter Gene Sequences (UCI ML Repository)
- 106 sequences, 57 nucleotides each
- 53 promoters, 53 non-promoters
- Binary classification task

## References

- Lodhi et al. "Text Classification using String Kernels" (2002)
- Leslie et al. "The Spectrum Kernel" (2002)
- SMO: Platt, "Sequential Minimal Optimization" (1998)
