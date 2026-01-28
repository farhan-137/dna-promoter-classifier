"""
DNA Promoter Sequence Classifier
================================
Classifies E. Coli DNA sequences as promoters or non-promoters
using a Support Vector Machine with String Kernels.

The cool part: instead of hand-crafting features, we use the kernel trick
to implicitly work in a 4^k dimensional space (k-mer frequencies).

Dataset: UCI ML Repository - E. Coli Promoter Gene Sequences
    - 106 sequences, 57 nucleotides each
    - 53 promoters (+), 53 non-promoters (-)

Usage:
    python train.py

@author: parzival
"""

import os
import time

from data_loader import load_promoter_data, train_test_split
from string_kernel import spectrum_kernel, normalized_spectrum_kernel, mismatch_kernel
from svm_scratch import StringKernelSVM, accuracy, confusion_matrix, precision_recall_f1


def print_header(text):
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def run_experiment(X_train, y_train, X_test, y_test, kernel_func, kernel_name, **kernel_params):
    """
    Train and evaluate SVM with a specific kernel configuration.
    """
    print(f"\n--- {kernel_name} ---")
    print(f"Kernel params: {kernel_params}")
    
    start = time.time()
    
    # create and train SVM
    svm = StringKernelSVM(
        kernel_func=kernel_func,
        C=1.0,
        tol=1e-3,
        max_iter=50,
        **kernel_params
    )
    
    svm.fit(X_train, y_train)
    train_time = time.time() - start
    
    # evaluate on training set
    train_preds = svm.predict(X_train)
    train_acc = accuracy(y_train, train_preds)
    
    # evaluate on test set
    test_preds = svm.predict(X_test)
    test_acc = accuracy(y_test, test_preds)
    
    # detailed metrics
    tp, tn, fp, fn = confusion_matrix(y_test, test_preds)
    prec, rec, f1 = precision_recall_f1(y_test, test_preds)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_acc * 100:.1f}%")
    print(f"  Test accuracy:     {test_acc * 100:.1f}%")
    print(f"  Training time:     {train_time:.2f}s")
    
    print(f"\nConfusion Matrix (test set):")
    print(f"                 Predicted")
    print(f"                 +     -")
    print(f"  Actual  +     {tp:3d}   {fn:3d}")
    print(f"          -     {fp:3d}   {tn:3d}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    return {
        'kernel': kernel_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'time': train_time,
        'f1': f1
    }


def main():
    print_header("DNA Promoter Sequence Classifier")
    print("Using String Kernels + SVM (implemented from scratch)")
    
    # load data
    data_path = os.path.join(os.path.dirname(__file__), "data", "promoters.data")
    
    print(f"\nLoading data from: {data_path}")
    sequences, labels, names = load_promoter_data(data_path)
    
    print(f"Total sequences: {len(sequences)}")
    print(f"Promoters (+1): {sum(1 for y in labels if y == 1)}")
    print(f"Non-promoters (-1): {sum(1 for y in labels if y == -1)}")
    print(f"Sequence length: {len(sequences[0])} nucleotides")
    
    # show a couple examples
    print(f"\nExample promoter: {sequences[0][:40]}...")
    print(f"Example non-prom: {sequences[-1][:40]}...")
    
    # split data
    X_train, y_train, X_test, y_test = train_test_split(
        sequences, labels, test_ratio=0.2, seed=42
    )
    
    print(f"\nTrain set: {len(X_train)} ({sum(1 for y in y_train if y == 1)} promoters)")
    print(f"Test set:  {len(X_test)} ({sum(1 for y in y_test if y == 1)} promoters)")
    
    # run experiments with different kernels
    print_header("Experiments")
    
    results = []
    
    # experiment 1: spectrum kernel with k=3
    r = run_experiment(
        X_train, y_train, X_test, y_test,
        spectrum_kernel, "Spectrum Kernel (k=3)", k=3
    )
    results.append(r)
    
    # experiment 2: spectrum kernel with k=4
    r = run_experiment(
        X_train, y_train, X_test, y_test,
        spectrum_kernel, "Spectrum Kernel (k=4)", k=4
    )
    results.append(r)
    
    # experiment 3: normalized spectrum kernel
    r = run_experiment(
        X_train, y_train, X_test, y_test,
        normalized_spectrum_kernel, "Normalized Spectrum (k=3)", k=3
    )
    results.append(r)
    
    # experiment 4: mismatch kernel (slower but more robust)
    r = run_experiment(
        X_train, y_train, X_test, y_test,
        mismatch_kernel, "Mismatch Kernel (k=3, m=1)", k=3, max_mismatch=1
    )
    results.append(r)
    
    # summary
    print_header("Summary")
    print(f"\n{'Kernel':<30} {'Train Acc':>10} {'Test Acc':>10} {'F1':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['kernel']:<30} {r['train_acc']*100:>9.1f}% {r['test_acc']*100:>9.1f}% {r['f1']:>8.3f} {r['time']:>7.2f}s")
    
    # best model
    best = max(results, key=lambda x: x['test_acc'])
    print(f"\nBest model: {best['kernel']} with {best['test_acc']*100:.1f}% test accuracy")
    
    print_header("Analysis")
    print("""
Key observations:
1. The spectrum kernel counts shared k-mers (substrings of length k).
   For DNA, k=3 gives 4^3 = 64 possible k-mers.
   For k=4, that's 256 features - but we never build the full vector!

2. The kernel trick lets us compute similarity in this high-dimensional
   space without explicitly creating feature vectors.

3. The mismatch kernel allows for biological variation (mutations)
   but is slower because it considers near-matches too.

4. Pre-computing the Gram matrix (all pairwise kernel values) is crucial
   for training speed. Without it, SMO would recompute kernels repeatedly.

This is why string kernels are powerful for bioinformatics:
- Proteins have 20 amino acids -> 20^k features for k-mers
- For k=5, that's 3.2 million features
- Kernel trick: we get the same result without building that vector
""")


if __name__ == "__main__":
    main()
