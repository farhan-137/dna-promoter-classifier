"""
Support Vector Machine with SMO (Sequential Minimal Optimization)
Built to work with string kernels for sequence classification.

SMO is basically: instead of solving the massive QP problem all at once,
we pick 2 alphas at a time and optimize just those. Repeat until converged.

@author: parzival
"""

import random
import math


class StringKernelSVM:
    """
    SVM classifier that works with any kernel function.
    
    Uses SMO algorithm for training - no scipy or cvxopt needed.
    The key insight: we precompute the gram matrix so kernel lookups are O(1).
    """
    
    def __init__(self, kernel_func=None, C=1.0, tol=1e-3, max_iter=100, **kernel_params):
        """
        Args:
            kernel_func: function(s1, s2, **params) -> similarity score
            C: regularization parameter (higher = less regularization)
            tol: numerical tolerance for convergence
            max_iter: max passes over the training data
            kernel_params: passed to kernel function (e.g., k=3 for spectrum)
        """
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        
        # these get set during training
        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.gram_matrix = None
        
        # support vectors (indices where alpha > 0)
        self.support_indices = []
    
    def _compute_gram_matrix(self, X):
        """
        Precompute all pairwise kernel values.
        
        For n training samples, this is n^2 kernel calculations.
        But we only do it once, and then SMO can look up values instantly.
        
        This is the "optimization" mentioned in the project description.
        Without this, SMO would recompute K(i,j) every time it's needed.
        """
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        
        print(f"Pre-computing gram matrix ({n}x{n} = {n*n} kernel evaluations)...")
        
        total = n * (n + 1) // 2  # upper triangle + diagonal
        computed = 0
        
        for i in range(n):
            for j in range(i, n):
                val = self.kernel_func(X[i], X[j], **self.kernel_params)
                K[i][j] = val
                K[j][i] = val  # symmetric
                computed += 1
            
            # progress every 25%
            if (i + 1) % max(1, n // 4) == 0:
                print(f"  ...{100 * (i+1) // n}% done")
        
        print("Gram matrix complete.")
        return K
    
    def _kernel(self, i, j):
        """Quick lookup in precomputed gram matrix."""
        return self.gram_matrix[i][j]
    
    def _decision_function(self, idx):
        """
        Compute f(x_idx) = sum_j(alpha_j * y_j * K(x_j, x_idx)) + b
        
        This is the raw SVM output before sign().
        """
        result = 0.0
        for j in range(len(self.X_train)):
            if self.alphas[j] > 0:  # only non-zero alphas matter
                result += self.alphas[j] * self.y_train[j] * self._kernel(j, idx)
        return result + self.b
    
    def _compute_error(self, idx):
        """E_idx = f(x_idx) - y_idx"""
        return self._decision_function(idx) - self.y_train[idx]
    
    def _take_step(self, i1, i2):
        """
        Optimize alphas[i1] and alphas[i2] together.
        This is the core of SMO - we can solve for 2 alphas analytically.
        
        Returns True if alphas were updated significantly.
        """
        if i1 == i2:
            return False
        
        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1 = self.y_train[i1]
        y2 = self.y_train[i2]
        
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)
        
        s = y1 * y2
        
        # compute bounds for alpha2
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L >= H:
            return False
        
        # second derivative of the objective (eta)
        k11 = self._kernel(i1, i1)
        k12 = self._kernel(i1, i2)
        k22 = self._kernel(i2, i2)
        eta = 2 * k12 - k11 - k22
        
        if eta >= 0:
            # rare edge case - skip this pair
            return False
        
        # compute new alpha2
        alpha2_new = alpha2_old - y2 * (E1 - E2) / eta
        
        # clip to bounds
        if alpha2_new > H:
            alpha2_new = H
        elif alpha2_new < L:
            alpha2_new = L
        
        # check if change is significant
        if abs(alpha2_new - alpha2_old) < self.tol * (alpha2_new + alpha2_old + self.tol):
            return False
        
        # compute new alpha1
        alpha1_new = alpha1_old + s * (alpha2_old - alpha2_new)
        
        # update threshold b
        b1 = self.b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
        b2 = self.b - E2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # store new alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new
        
        return True
    
    def _examine_example(self, i2):
        """
        Try to find a good i1 to pair with i2 for optimization.
        Returns True if we made progress.
        """
        y2 = self.y_train[i2]
        alpha2 = self.alphas[i2]
        E2 = self._compute_error(i2)
        r2 = E2 * y2
        
        # check KKT conditions
        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            # this point violates KKT - try to optimize it
            
            # first: try the one with maximum |E1 - E2|
            # this gives the biggest step
            non_bound = [i for i in range(len(self.alphas)) if 0 < self.alphas[i] < self.C]
            
            if len(non_bound) > 1:
                # find i1 that maximizes |E1 - E2|
                max_delta = 0
                i1_best = -1
                for i1 in non_bound:
                    E1 = self._compute_error(i1)
                    delta = abs(E1 - E2)
                    if delta > max_delta:
                        max_delta = delta
                        i1_best = i1
                
                if i1_best >= 0 and self._take_step(i1_best, i2):
                    return True
            
            # second: loop over non-bound alphas (random start)
            start = random.randint(0, len(non_bound) - 1) if non_bound else 0
            for i in range(len(non_bound)):
                i1 = non_bound[(start + i) % len(non_bound)]
                if self._take_step(i1, i2):
                    return True
            
            # third: loop over all alphas
            n = len(self.alphas)
            start = random.randint(0, n - 1)
            for i in range(n):
                i1 = (start + i) % n
                if self._take_step(i1, i2):
                    return True
        
        return False
    
    def fit(self, X, y):
        """
        Train the SVM on sequences X with labels y.
        
        Args:
            X: list of sequences (strings)
            y: list of labels (+1 or -1)
        """
        self.X_train = X
        self.y_train = y
        n = len(X)
        
        # initialize alphas to zero
        self.alphas = [0.0] * n
        self.b = 0.0
        
        # precompute gram matrix (the key optimization!)
        self.gram_matrix = self._compute_gram_matrix(X)
        
        print(f"\nTraining SVM with SMO (C={self.C}, max_iter={self.max_iter})...")
        
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # loop over all training examples
                for i in range(n):
                    if self._examine_example(i):
                        num_changed += 1
            else:
                # loop over examples where 0 < alpha < C
                for i in range(n):
                    if 0 < self.alphas[i] < self.C:
                        if self._examine_example(i):
                            num_changed += 1
            
            iteration += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            # progress
            n_sv = sum(1 for a in self.alphas if a > 0)
            if iteration % 10 == 0 or iteration < 5:
                print(f"  Iter {iteration}: {num_changed} alphas changed, {n_sv} support vectors")
        
        # identify support vectors
        self.support_indices = [i for i in range(n) if self.alphas[i] > 1e-6]
        print(f"\nTraining complete. {len(self.support_indices)} support vectors found.")
        
        return self
    
    def predict_one(self, x_new):
        """
        Predict class for a single new sequence.
        
        Returns: +1 or -1
        """
        # compute kernel with all support vectors
        result = 0.0
        for i in self.support_indices:
            k_val = self.kernel_func(x_new, self.X_train[i], **self.kernel_params)
            result += self.alphas[i] * self.y_train[i] * k_val
        result += self.b
        
        return 1 if result >= 0 else -1
    
    def predict(self, X_test):
        """Predict classes for multiple sequences."""
        return [self.predict_one(x) for x in X_test]
    
    def decision_scores(self, X_test):
        """Get raw scores (before sign) for each test sample."""
        scores = []
        for x in X_test:
            result = 0.0
            for i in self.support_indices:
                k_val = self.kernel_func(x, self.X_train[i], **self.kernel_params)
                result += self.alphas[i] * self.y_train[i] * k_val
            result += self.b
            scores.append(result)
        return scores


def accuracy(y_true, y_pred):
    """Calculate accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred):
    """
    Returns: (TP, TN, FP, FN)
    Assumes +1 is positive class.
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == -1)
    return tp, tn, fp, fn


def precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall, and F1 score."""
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


if __name__ == "__main__":
    # quick test with a simple kernel
    from string_kernel import spectrum_kernel
    
    # toy data
    X = ["aaaa", "aaat", "tttt", "ttta"]
    y = [1, 1, -1, -1]
    
    print("Testing SVM with toy data...")
    svm = StringKernelSVM(kernel_func=spectrum_kernel, C=1.0, k=2)
    svm.fit(X, y)
    
    preds = svm.predict(X)
    print(f"Predictions: {preds}")
    print(f"Actual:      {y}")
    print(f"Accuracy: {accuracy(y, preds):.2f}")
