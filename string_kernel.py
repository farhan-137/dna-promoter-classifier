"""
String Kernel implementations for DNA/protein sequence classification.

Two kernels:
1. Spectrum kernel - counts shared k-mers (substrings of length k)
2. Subsequence kernel - counts shared subsequences with gap penalty

The kernel trick: we compute similarity without building the full 4^k feature vector.
For k=5, that's 1024 features. For k=10, over 1 million. But we only need the dot product.

@author: parzival
"""

from collections import Counter


def get_kmers(sequence, k):
    """
    Extract all k-mers (substrings of length k) from a sequence.
    
    Example:
        get_kmers("acgt", 2) -> ["ac", "cg", "gt"]
    """
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def spectrum_kernel(s1, s2, k=3):
    """
    Spectrum kernel: counts shared k-mers between two sequences.
    
    This is the dot product of k-mer frequency vectors, but computed
    without explicitly building those massive vectors.
    
    For DNA with k=5: 4^5 = 1024 possible k-mers
    For proteins with k=5: 20^5 = 3.2 million possible k-mers
    
    Args:
        s1, s2: input sequences (strings)
        k: k-mer length
    
    Returns:
        similarity score (higher = more similar)
    """
    # extract k-mers
    kmers1 = get_kmers(s1, k)
    kmers2 = get_kmers(s2, k)
    
    if not kmers1 or not kmers2:
        return 0.0
    
    # count frequencies
    counts1 = Counter(kmers1)
    counts2 = Counter(kmers2)
    
    # dot product: sum of count1[kmer] * count2[kmer] for shared kmers
    score = 0.0
    for kmer, cnt1 in counts1.items():
        if kmer in counts2:
            score += cnt1 * counts2[kmer]
    
    return score


def normalized_spectrum_kernel(s1, s2, k=3):
    """
    Normalized spectrum kernel.
    K_norm(s1, s2) = K(s1, s2) / sqrt(K(s1, s1) * K(s2, s2))
    
    This gives values between 0 and 1, which can help with training stability.
    """
    k12 = spectrum_kernel(s1, s2, k)
    k11 = spectrum_kernel(s1, s1, k)
    k22 = spectrum_kernel(s2, s2, k)
    
    if k11 == 0 or k22 == 0:
        return 0.0
    
    return k12 / ((k11 * k22) ** 0.5)


def subsequence_kernel(s1, s2, subseq_len=3, decay=0.8):
    """
    Subsequence kernel with gap penalty.
    
    Unlike spectrum kernel, this allows gaps between characters.
    The decay factor penalizes longer gaps.
    
    Example:
        s1 = "cat"
        s2 = "cart" 
        The subsequence "c-t" exists in both (with gap in s2)
        Score is weighted by decay^(gap_length)
    
    Uses dynamic programming - O(n*m*subseq_len) time complexity.
    
    Args:
        s1, s2: input sequences
        subseq_len: length of subsequences to consider (usually 2-4)
        decay: gap penalty factor (0 < decay <= 1)
    """
    n = len(s1)
    m = len(s2)
    
    if n == 0 or m == 0:
        return 0.0
    
    # DP tables
    # K[i][j] = kernel value using s1[:i] and s2[:j]
    # K_prime[l][i][j] = intermediate values for subsequence of length l
    
    # we'll use a simpler approach: count matching subsequences
    # with decay based on span length
    
    K = [[0.0] * (m + 1) for _ in range(n + 1)]
    K_prime = [[[0.0] * (m + 1) for _ in range(n + 1)] for _ in range(subseq_len + 1)]
    
    # base case: empty subsequence
    for i in range(n + 1):
        for j in range(m + 1):
            K_prime[0][i][j] = 1.0
    
    # fill DP tables
    for l in range(1, subseq_len + 1):
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # decay from previous positions
                K_prime[l][i][j] = decay * K_prime[l][i][j-1]
                
                if s1[i-1] == s2[j-1]:
                    K_prime[l][i][j] += decay * K_prime[l-1][i-1][j-1]
                
                K_prime[l][i][j] += decay * K_prime[l][i-1][j] - (decay ** 2) * K_prime[l][i-1][j-1]
    
    # final kernel value
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                K[i][j] = K[i-1][j-1] + decay * decay * K_prime[subseq_len-1][i-1][j-1]
            else:
                K[i][j] = K[i-1][j] + K[i][j-1] - K[i-1][j-1]
    
    return K[n][m]


def mismatch_kernel(s1, s2, k=3, max_mismatch=1):
    """
    Mismatch kernel: like spectrum but allows some mismatches.
    
    A k-mer from s1 matches a k-mer from s2 if they differ by at most
    max_mismatch positions. This is more robust to mutations.
    
    Slower than spectrum but handles sequencing errors better.
    """
    kmers1 = get_kmers(s1, k)
    kmers2 = get_kmers(s2, k)
    
    if not kmers1 or not kmers2:
        return 0.0
    
    def hamming_distance(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b))
    
    # count matches with allowed mismatches
    score = 0.0
    for km1 in kmers1:
        for km2 in kmers2:
            if hamming_distance(km1, km2) <= max_mismatch:
                score += 1.0
    
    return score


# Precomputed kernel class for efficiency
class PrecomputedKernel:
    """
    Compute gram matrix once, then look up values.
    This avoids recomputing string kernels during SMO training.
    """
    
    def __init__(self, kernel_func, **kernel_params):
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.gram_matrix = None
        self.sequences = None
    
    def build_gram_matrix(self, sequences):
        """
        Compute K[i][j] for all pairs of training sequences.
        This is O(n^2 * kernel_cost) but only done once.
        """
        n = len(sequences)
        self.sequences = sequences
        self.gram_matrix = [[0.0] * n for _ in range(n)]
        
        # gram matrix is symmetric, so only compute upper triangle
        for i in range(n):
            for j in range(i, n):
                val = self.kernel_func(sequences[i], sequences[j], **self.kernel_params)
                self.gram_matrix[i][j] = val
                self.gram_matrix[j][i] = val
            
            # progress indicator for large datasets
            if (i + 1) % 20 == 0:
                print(f"  Gram matrix: {i+1}/{n} rows computed")
        
        return self.gram_matrix
    
    def get(self, i, j):
        """Get precomputed kernel value."""
        return self.gram_matrix[i][j]
    
    def compute_new(self, s_new, s_train_idx):
        """
        Compute kernel between a new sequence and a training sequence.
        Used during prediction.
        """
        return self.kernel_func(s_new, self.sequences[s_train_idx], **self.kernel_params)


if __name__ == "__main__":
    # test the kernels
    print("String Kernel Tests")
    print("=" * 40)
    
    # test sequences
    s1 = "acgtacgt"
    s2 = "acgtacgt"  # identical
    s3 = "tgcatgca"  # different
    s4 = "acgttcgt"  # similar (1 change)
    
    print(f"\ns1 = {s1}")
    print(f"s2 = {s2}")
    print(f"s3 = {s3}")
    print(f"s4 = {s4}")
    
    print("\n--- Spectrum Kernel (k=3) ---")
    print(f"K(s1, s2) = {spectrum_kernel(s1, s2, k=3):.2f}  (identical)")
    print(f"K(s1, s3) = {spectrum_kernel(s1, s3, k=3):.2f}  (different)")
    print(f"K(s1, s4) = {spectrum_kernel(s1, s4, k=3):.2f}  (similar)")
    
    print("\n--- Normalized Spectrum Kernel (k=3) ---")
    print(f"K_norm(s1, s2) = {normalized_spectrum_kernel(s1, s2, k=3):.4f}")
    print(f"K_norm(s1, s3) = {normalized_spectrum_kernel(s1, s3, k=3):.4f}")
    print(f"K_norm(s1, s4) = {normalized_spectrum_kernel(s1, s4, k=3):.4f}")
    
    print("\n--- Mismatch Kernel (k=3, m=1) ---")
    print(f"K(s1, s2) = {mismatch_kernel(s1, s2, k=3):.2f}")
    print(f"K(s1, s3) = {mismatch_kernel(s1, s3, k=3):.2f}")
    print(f"K(s1, s4) = {mismatch_kernel(s1, s4, k=3):.2f}")
    
    # sanity check: kernel with itself should be maximal
    print("\nSanity check: K(s, s) should be maximal")
    k_self = spectrum_kernel(s1, s1, k=3)
    k_other = spectrum_kernel(s1, s3, k=3)
    print(f"K(s1, s1) = {k_self:.2f} > K(s1, s3) = {k_other:.2f} ? {k_self > k_other}")
