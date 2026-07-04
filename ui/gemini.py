import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------- Activations ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0.01)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

# ---------- True RFCS Unit ----------
class RFCSUnit:
    """
    A single RFCS Unit with out-degree ≤ 3
    Feature complement = aggregation of OTHER features
    """
    def __init__(self, input_size, output_size=3, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size  # k = 3 out-degree
        
        # Initialize weights
        scale = np.sqrt(2.0 / input_size)
        self.W1 = np.random.randn(output_size, input_size) * scale  # complement weights
        self.W2 = np.random.randn(output_size, input_size) * scale  # fusion weights
        self.scale = np.random.randn(output_size) * 0.1 + 1.0
        self.bias = np.random.randn(output_size) * 0.1
        
        # ADAM state
        self.m = {
            'W1': np.zeros_like(self.W1),
            'W2': np.zeros_like(self.W2),
            'scale': np.zeros_like(self.scale),
            'bias': np.zeros_like(self.bias)
        }
        self.v = {
            'W1': np.zeros_like(self.W1),
            'W2': np.zeros_like(self.W2),
            'scale': np.zeros_like(self.scale),
            'bias': np.zeros_like(self.bias)
        }
        self.t = 0
        
        self.activation = activation
        if activation == 'relu':
            self.act = relu
            self.d_act = d_relu
        elif activation == 'tanh':
            self.act = tanh
            self.d_act = d_tanh
        else:
            self.act = sigmoid
            self.d_act = lambda x: sigmoid(x) * (1 - sigmoid(x))
    
    def forward(self, x):
        """
        x: input vector of shape (batch_size, input_size)
        Returns: output vector of shape (batch_size, output_size)
        """
        batch_size = x.shape[0]
        
        # For each feature i, complement = aggregation of OTHER features
        # Aggregation: arithmetic mean of other features
        complements = []
        for i in range(self.input_size):
            # Create mask for all features except i
            mask = np.ones(self.input_size, dtype=bool)
            mask[i] = False
            # Mean of other features
            comp = np.mean(x[:, mask], axis=1, keepdims=True)
            complements.append(comp)
        complements = np.concatenate(complements, axis=1)  # (batch, input_size)
        
        # Compute fusion for each output neuron
        outputs = []
        for k in range(self.output_size):
            # Each output neuron combines ALL input features with their complements
            # Using weight vectors W1[k] and W2[k]
            combined = []
            for i in range(self.input_size):
                # For feature i: combine raw x_i with its complement
                fusion_input = self.W1[k, i] * x[:, i:i+1] + self.W2[k, i] * complements[:, i:i+1]
                combined.append(fusion_input)
            combined = np.concatenate(combined, axis=1)  # (batch, input_size)
            
            # Sum across features and apply activation
            z = np.sum(combined, axis=1, keepdims=True) * self.scale[k] + self.bias[k]
            out = self.act(z)
            outputs.append(out)
        
        return np.concatenate(outputs, axis=1)  # (batch, output_size)
    
    def backward(self, x, grad_output):
        """
        x: input to this unit
        grad_output: gradient from next layer (batch, output_size)
        Returns: gradients for weights and grad_input
        """
        batch_size = x.shape[0]
        
        # Compute complements (same as forward)
        complements = []
        for i in range(self.input_size):
            mask = np.ones(self.input_size, dtype=bool)
            mask[i] = False
            comp = np.mean(x[:, mask], axis=1, keepdims=True)
            complements.append(comp)
        complements = np.concatenate(complements, axis=1)
        
        # Gradients for each output neuron
        grad_W1 = np.zeros_like(self.W1)
        grad_W2 = np.zeros_like(self.W2)
        grad_scale = np.zeros_like(self.scale)
        grad_bias = np.zeros_like(self.bias)
        grad_x = np.zeros_like(x)
        
        for k in range(self.output_size):
            # Combined input to this output neuron
            combined = []
            for i in range(self.input_size):
                fusion_input = self.W1[k, i] * x[:, i:i+1] + self.W2[k, i] * complements[:, i:i+1]
                combined.append(fusion_input)
            combined = np.concatenate(combined, axis=1)
            
            z = np.sum(combined, axis=1, keepdims=True) * self.scale[k] + self.bias[k]
            d_act = self.d_act(z) * grad_output[:, k:k+1]
            
            # Gradients for scale and bias
            grad_scale[k] = np.sum(d_act * np.sum(combined, axis=1, keepdims=True))
            grad_bias[k] = np.sum(d_act)
            
            # Gradients for W1 and W2
            d_combined = d_act * self.scale[k]
            for i in range(self.input_size):
                grad_W1[k, i] = np.sum(d_combined * x[:, i:i+1])
                grad_W2[k, i] = np.sum(d_combined * complements[:, i:i+1])
                
                # Gradient to input x_i
                # Since complement depends on other features, we need to distribute gradient
                # This is the gradient through W1 directly
                grad_x[:, i:i+1] += d_combined * self.W1[k, i]
                
                # Gradient through complement (mean of other features)
                # For complement of feature i, we used mean of all other features
                # So gradient to each other feature j is grad_complement / (input_size - 1)
                for j in range(self.input_size):
                    if j != i:
                        grad_x[:, j:j+1] += d_combined * self.W2[k, i] / (self.input_size - 1)
        
        # Clip gradients
        clip_value = 5.0
        grad_W1 = np.clip(grad_W1, -clip_value, clip_value)
        grad_W2 = np.clip(grad_W2, -clip_value, clip_value)
        grad_scale = np.clip(grad_scale, -clip_value, clip_value)
        grad_bias = np.clip(grad_bias, -clip_value, clip_value)
        grad_x = np.clip(grad_x, -clip_value, clip_value)
        
        return {
            'grad_W1': grad_W1,
            'grad_W2': grad_W2,
            'grad_scale': grad_scale,
            'grad_bias': grad_bias,
            'grad_x': grad_x
        }
    
    def update(self, grads, lr):
        """ADAM update"""
        self.t += 1
        
        for param in ['W1', 'W2', 'scale', 'bias']:
            grad = grads[f'grad_{param}']
            
            # ADAM update
            self.m[param] = 0.9 * self.m[param] + 0.1 * grad
            self.v[param] = 0.999 * self.v[param] + 0.001 * (grad ** 2)
            
            m_hat = self.m[param] / (1 - 0.9 ** self.t)
            v_hat = self.v[param] / (1 - 0.999 ** self.t)
            
            if param == 'W1':
                self.W1 -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif param == 'W2':
                self.W2 -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif param == 'scale':
                self.scale -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif param == 'bias':
                self.bias -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

# ---------- RFCS Network (Stack of Units) ----------
class RFCSNetwork:
    """
    Stack of RFCS Units with sequential pipeline
    Each unit has out-degree = 3 (k=3)
    """
    def __init__(self, input_size, hidden_sizes=[3, 3, 3], output_size=1, lr=0.01):
        self.lr = lr
        self.units = []
        
        # Build sequential pipeline of units
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.units.append(RFCSUnit(prev_size, hidden_size, activation='relu'))
            prev_size = hidden_size
        
        # Output layer (linear)
        scale = np.sqrt(2.0 / prev_size)
        self.W_out = np.random.randn(output_size, prev_size) * scale
        self.b_out = np.random.randn(output_size) * 0.1
        
        # ADAM state for output layer
        self.mW_out = np.zeros_like(self.W_out)
        self.vW_out = np.zeros_like(self.W_out)
        self.mb_out = np.zeros_like(self.b_out)
        self.vb_out = np.zeros_like(self.b_out)
        self.t = 0
        
        self.loss_history = []
        self.acc_history = []
    
    def forward(self, x):
        """Forward pass through all units"""
        current = x
        for unit in self.units:
            current = unit.forward(current)
        
        # Output layer
        logits = current @ self.W_out.T + self.b_out
        logits = np.clip(logits, -10, 10)
        probs = sigmoid(logits)
        
        return {
            'features': current,
            'logits': logits,
            'probs': probs
        }
    
    def backward(self, x, y):
        """Backward pass through all units"""
        batch_size = x.shape[0]
        cache = self.forward(x)
        probs = cache['probs']
        
        # Loss
        loss = -np.mean(y * np.log(probs + 1e-9) + (1-y) * np.log(1 - probs + 1e-9))
        
        # Gradient of loss w.r.t logits
        d_logits = (probs - y) / batch_size
        
        # Gradient of output layer
        dW_out = d_logits.T @ cache['features']
        db_out = np.sum(d_logits, axis=0, keepdims=True)
        
        # Gradient to previous layer
        grad_input = d_logits @ self.W_out
        
        # Backprop through units in reverse
        unit_grads = []
        for unit in reversed(self.units):
            grad = unit.backward(x, grad_input)
            unit_grads.append(grad)
            grad_input = grad['grad_x']
        
        # Clip gradients
        clip_value = 5.0
        dW_out = np.clip(dW_out, -clip_value, clip_value)
        db_out = np.clip(db_out, -clip_value, clip_value)
        
        return {
            'loss': loss,
            'dW_out': dW_out,
            'db_out': db_out,
            'unit_grads': list(reversed(unit_grads))
        }
    
    def step(self, x, y):
        """ADAM update step"""
        grads = self.backward(x, y)
        self.t += 1
        
        # Update output layer with ADAM
        self.mW_out = 0.9 * self.mW_out + 0.1 * grads['dW_out']
        self.vW_out = 0.999 * self.vW_out + 0.001 * (grads['dW_out'] ** 2)
        mW_hat = self.mW_out / (1 - 0.9 ** self.t)
        vW_hat = self.vW_out / (1 - 0.999 ** self.t)
        self.W_out -= self.lr * mW_hat / (np.sqrt(vW_hat) + 1e-8)
        
        self.mb_out = 0.9 * self.mb_out + 0.1 * grads['db_out']
        self.vb_out = 0.999 * self.vb_out + 0.001 * (grads['db_out'] ** 2)
        mb_hat = self.mb_out / (1 - 0.9 ** self.t)
        vb_hat = self.vb_out / (1 - 0.999 ** self.t)
        self.b_out -= self.lr * mb_hat / (np.sqrt(vb_hat) + 1e-8)
        
        # Update each unit
        for i, unit in enumerate(self.units):
            unit.update(grads['unit_grads'][i], self.lr)
        
        return grads['loss']
    
    def predict(self, x):
        """Predict probabilities"""
        return self.forward(x)['probs']
    
    def predict_classes(self, x):
        """Predict class labels"""
        return (self.predict(x) >= 0.5).astype(int).flatten()
    
    def accuracy(self, x, y):
        """Compute accuracy"""
        preds = self.predict_classes(x)
        return np.mean(preds == y.flatten())
    
    def fit(self, x, y, epochs=300, batch_size=32, verbose=True):
        """Train the model"""
        n = x.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for i in range(0, n, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                loss = self.step(x_batch, y_batch)
                epoch_loss += loss * len(x_batch)
            
            avg_loss = epoch_loss / n
            acc = self.accuracy(x, y)
            
            self.loss_history.append(avg_loss)
            self.acc_history.append(acc)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}")
            
            # Early stopping
            if avg_loss < 0.05:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self

# ---------- Visualization ----------
def plot_decision_boundary(model, X, y, title="RFCS Decision Boundary"):
    """Plot decision boundary and data points"""
    # Create grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Decision boundary
    ax1 = axes[0]
    ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(title)
    
    # Loss and accuracy curves
    ax2 = axes[1]
    epochs = range(1, len(model.loss_history) + 1)
    ax2.plot(epochs, model.loss_history, 'b-', label='Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax3 = ax2.twinx()
    ax3.plot(epochs, model.acc_history, 'r-', label='Accuracy', linewidth=2)
    ax3.set_ylabel('Accuracy', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Training Progress')
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.show()

# ---------- Main Training ----------
def main():
    print("=" * 70)
    print("True RFCS (Repeated Complementary Feature Strands) - Moons Dataset")
    print("Each unit has out-degree k=3 (max 3 out-edges)")
    print("=" * 70)
    
    # Generate moons dataset
    X, y = make_moons(n_samples=500, noise=0.15, random_state=42)
    y = y.reshape(-1, 1)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"RFCS Units: 3 units, each with out-degree=3")
    print(f"Total parameters: ~{(3*2*2 + 3 + 3*2 + 1)}")  # Rough estimate
    print("\nTraining...")
    
    # Create and train model
    # 3 units: first expands to 3 features, second to 3, third to 3
    model = RFCSNetwork(
        input_size=2,
        hidden_sizes=[3, 3, 3],  # Each unit has out-degree=3 (k=3)
        output_size=1,
        lr=0.01
    )
    model.fit(X_train_scaled, y_train, epochs=300, batch_size=32)
    
    # Evaluate
    train_acc = model.accuracy(X_train_scaled, y_train)
    test_acc = model.accuracy(X_test_scaled, y_test)
    
    print("\n" + "=" * 70)
    print("Results:")
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 70)
    
    print("\n🔬 RFCS Architecture:")
    print("  Input (2 features)")
    for i, unit in enumerate(model.units):
        print(f"  ↓ Unit {i+1} (out-degree k=3) → {unit.output_size} features")
    print(f"  ↓ Output (1 neuron)")
    
    # Visualize
    plot_decision_boundary(model, X_test_scaled, y_test.flatten(), 
                          f"RFCS Decision Boundary (Test Acc: {test_acc*100:.1f}%)")
    
    return model

if __name__ == "__main__":
    model = main()