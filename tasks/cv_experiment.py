"""
RFCS (Repeated Complementary Feature Strands) Classifier
Fast implementation using vectorized NumPy with autograd via finite differences.
15 total parameters for DIM=2, R=1, 1 perspective.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Union
import warnings


class RFCSClassifier(BaseEstimator, ClassifierMixin):
    """
    Fast RFCS Classifier with vectorized NumPy operations.
    
    Parameter breakdown (DIM=2, R=1, 1 perspective):
    - Per thought (2 thoughts): w1, w2, scale, bias, cW1, cW2, cB = 7 × 2 = 14
    - Shared elaboration: pW = 1
    Total: 15 parameters
    
    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    
    learning_rate : float, default=0.01
        Learning rate for Adam optimizer.
    
    batch_size : int, default=None
        Batch size for mini-batch training. If None, use full batch.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    tol : float, default=1e-6
        Tolerance for optimization stopping criterion.
    
    verbose : bool, default=False
        Whether to print progress during fitting.
    
    solver : str, default='adam'
        Optimization solver. Currently only 'adam' is supported.
    
    Attributes
    ----------
    params_ : np.ndarray of shape (15,)
        Learned parameters of the model.
    
    classes_ : np.ndarray
        Unique class labels.
    
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        random_state: Optional[int] = None,
        tol: float = 1e-6,
        verbose: bool = False,
        solver: str = 'adam'  # Added back for compatibility
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.solver = solver  # Store for compatibility
        
        # Fixed architecture
        self.DIM = 2
        self.R = 1
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with clipping for numerical stability."""
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _init_params(self) -> np.ndarray:
        """Initialize 15 parameters with Xavier-like initialization."""
        rng = np.random.RandomState(self.random_state)
        
        params = []
        
        # Thought 0 (7 parameters)
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # w1_0
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # w2_0
        params.extend(rng.uniform(0.3, 0.7, 1).tolist())   # scale_0
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # bias_0
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cW1_0
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cW2_0
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cB_0
        
        # Thought 1 (7 parameters)
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # w1_1
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # w2_1
        params.extend(rng.uniform(0.3, 0.7, 1).tolist())   # scale_1
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # bias_1
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cW1_1
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cW2_1
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # cB_1
        
        # Shared elaboration weight (1 parameter)
        params.extend(rng.uniform(-0.5, 0.5, 1).tolist())  # pW
        
        return np.array(params, dtype=np.float32)
    
    def _unpack_params(self, params: np.ndarray) -> dict:
        """Unpack parameters for vectorized operations."""
        idx = 0
        
        # Thought 0
        w1_0 = params[idx]; idx += 1
        w2_0 = params[idx]; idx += 1
        scale_0 = params[idx]; idx += 1
        bias_0 = params[idx]; idx += 1
        cW1_0 = params[idx]; idx += 1
        cW2_0 = params[idx]; idx += 1
        cB_0 = params[idx]; idx += 1
        
        # Thought 1
        w1_1 = params[idx]; idx += 1
        w2_1 = params[idx]; idx += 1
        scale_1 = params[idx]; idx += 1
        bias_1 = params[idx]; idx += 1
        cW1_1 = params[idx]; idx += 1
        cW2_1 = params[idx]; idx += 1
        cB_1 = params[idx]; idx += 1
        
        # Shared elaboration weight
        pW = params[idx]
        
        return {
            'w1': np.array([w1_0, w1_1]),
            'w2': np.array([w2_0, w2_1]),
            'scale': np.array([scale_0, scale_1]),
            'bias': np.array([bias_0, bias_1]),
            'cW1': np.array([cW1_0, cW1_1]),
            'cW2': np.array([cW2_0, cW2_1]),
            'cB': np.array([cB_0, cB_1]),
            'pW': pW
        }
    
    def _forward_vectorized(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Vectorized forward pass for all samples.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, 2)
            Input features.
        params : np.ndarray of shape (15,)
            Model parameters.
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Prediction probabilities for class 1.
        """
        # Unpack parameters
        idx = 0
        w1_0 = params[idx]; idx += 1
        w2_0 = params[idx]; idx += 1
        scale_0 = params[idx]; idx += 1
        bias_0 = params[idx]; idx += 1
        cW1_0 = params[idx]; idx += 1
        cW2_0 = params[idx]; idx += 1
        cB_0 = params[idx]; idx += 1
        
        w1_1 = params[idx]; idx += 1
        w2_1 = params[idx]; idx += 1
        scale_1 = params[idx]; idx += 1
        bias_1 = params[idx]; idx += 1
        cW1_1 = params[idx]; idx += 1
        cW2_1 = params[idx]; idx += 1
        cB_1 = params[idx]; idx += 1
        
        pW = params[idx]
        
        # Vectorized operations for all samples
        x0 = X[:, 0]
        x1 = X[:, 1]
        
        # Initial normalized inputs (with interaction feature)
        cur0 = self._sigmoid(x0)
        cur1 = self._sigmoid(x1)
        cur2 = self._sigmoid(x0 * x1)
        
        # R=1 repetition
        total = cur0 + cur1 + cur2
        comp0 = (total - cur0) / 1.0  # DIM - 1 = 1
        comp1 = (total - cur1) / 1.0
        
        # Fusion for thought 0
        comb0 = cur0 * w1_0 + comp0 * w2_0
        fus0 = scale_0 * comb0 + bias_0
        grid0 = self._sigmoid(fus0)
        
        # Fusion for thought 1
        comb1 = cur1 * w1_1 + comp1 * w2_1
        fus1 = scale_1 * comb1 + bias_1
        grid1 = self._sigmoid(fus1)
        
        # Elaboration: derived from grid
        stacked0 = self._sigmoid(grid0 * cW1_0 + grid1 * cW2_0 + cB_0)
        stacked1 = self._sigmoid(grid1 * cW1_1 + grid0 * cW2_1 + cB_1)
        
        # Product and 1 perspective (paragraph aggregation)
        s2s = grid0 * stacked0 + grid1 * stacked1
        act = self._sigmoid(pW * s2s)
        
        return act
    
    def _loss(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        pred = self._forward_vectorized(X, params)
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    
    def _gradient(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient via finite differences (15 params only, so fast).
        Uses central difference for better accuracy.
        """
        eps = 1e-6
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            loss_plus = self._loss(params_plus, X, y)
            loss_minus = self._loss(params_minus, X, y)
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the RFCS classifier to the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must have exactly 2 features.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Ensure 2 features
        if X.shape[1] != 2:
            raise ValueError(f"RFCS requires exactly 2 features, got {X.shape[1]}")
        
        # Store classes
        self.classes_ = unique_labels(y)
        
        # Convert to binary (0, 1)
        if len(self.classes_) > 2:
            raise ValueError("RFCS only supports binary classification")
        
        y_binary = (y == self.classes_[1]).astype(np.float32)
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize parameters
        params = self._init_params()
        
        # Adam optimizer
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        n_samples = len(X)
        batch_size = self.batch_size or n_samples
        
        best_loss = float('inf')
        best_params = params.copy()
        
        for iteration in range(self.max_iter):
            # Mini-batch
            if batch_size < n_samples:
                indices = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[indices]
                y_batch = y_binary[indices]
            else:
                X_batch = X
                y_batch = y_binary
            
            # Compute gradient
            grad = self._gradient(params, X_batch, y_batch)
            
            # Adam update
            t = iteration + 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            
            # Track best
            loss = self._loss(params, X, y_binary)
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.6f}")
            
            # Check convergence
            if np.max(np.abs(grad)) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        self.params_ = best_params
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probability of each class.
        """
        check_is_fitted(self, 'params_')
        X = check_array(X, accept_sparse=False)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        prob_class1 = self._forward_vectorized(X, self.params_)
        prob_class1 = np.clip(prob_class1, 0, 1)
        
        proba = np.column_stack([1 - prob_class1, prob_class1])
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return mean accuracy on the given test data and labels.
        """
        return super().score(X, y)
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator."""
        return {
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'random_state': self.random_state,
            'tol': self.tol,
            'verbose': self.verbose,
            'solver': self.solver  # Added for compatibility
        }
    
    def set_params(self, **params) -> 'RFCSClassifier':
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    import time
    
    print("=== RFCS Classifier Test ===")
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RFCSClassifier(
            max_iter=500,
            learning_rate=0.01,
            random_state=42,
            verbose=False,
            solver='adam'  # Now works!
        ))
    ])
    
    # Fit and evaluate
    print("Training RFCS classifier...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Accuracy: {acc:.4f}")
    print(f"Parameters: {len(pipeline.named_steps['model'].params_)}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    print("\nCross-validation scores:")
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")