import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import warnings

def load_clf_2d(name, n_samples=1000, noise=0.0, random_state=42, **kwargs):
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)
    
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        
    elif name == 'blobs':
        n_clusters = kwargs.get('n_clusters', 2)
        X, y = make_blobs(n_samples=n_samples, n_features=2, 
                         centers=n_clusters, cluster_std=0.3,  # Tight clusters
                         random_state=random_state)
        
    elif name == 'circles':
        factor = kwargs.get('factor', 0.3)  # Tighter concentric circles
        X, y = make_circles(n_samples=n_samples, noise=noise, 
                           factor=factor, random_state=random_state)
        
    elif name == 'spiral':
        X, y = make_spiral(n_samples, noise, random_state, **kwargs)
        
    elif name == 'xor':
        X, y = make_xor(n_samples, noise, random_state)
        
    elif name == 'checkerboard':
        X, y = make_checkerboard(n_samples, noise, random_state, **kwargs)
        
    elif name == 'stripes':
        X, y = make_stripes(n_samples, noise, random_state, **kwargs)
        
    elif name == 'perpendicular':
        X, y = make_perpendicular(n_samples, noise, random_state)
        
    elif name == 'x-mark':
        X, y = make_x_mark(n_samples, noise, random_state)
        
    elif name == 'stars':
        X, y = make_stars(n_samples, noise, random_state, **kwargs)
        
    elif name in ['triangle', 'square', 'pentagon', 'hexagon', 'heptagon', 'octagon']:
        n_sides = {'triangle': 3, 'square': 4, 'pentagon': 5,
                  'hexagon': 6, 'heptagon': 7, 'octagon': 8}[name]
        X, y = make_concentric_polygons(n_samples, n_sides, noise, random_state, **kwargs)
        
    elif name == 'concentric_polygons':
        n_sides = kwargs.get('n_sides', 6)
        X, y = make_concentric_polygons(n_samples, n_sides, noise, random_state, **kwargs)
        
    elif name == 'diamond':
        X, y = make_diamond(n_samples, noise, random_state)
        
    elif name == 'cross':
        X, y = make_cross(n_samples, noise, random_state)
        
    elif name == 's-curve':
        X, y = make_s_curve(n_samples, noise, random_state)
        
    elif name == 'two_diagonals':
        X, y = make_two_diagonals(n_samples, noise, random_state)
        
    elif name == 'ring':
        X, y = make_ring(n_samples, noise, random_state, **kwargs)
        
    elif name == 'grid':
        X, y = make_grid(n_samples, noise, random_state, **kwargs)
        
    elif name == 'swiss_roll_2d':
        X, y = make_swiss_roll_2d(n_samples, noise, random_state)
        
    else:
        raise ValueError(f"Unknown dataset name: {name}. "
                        f"Available: moons, blobs, circles, spiral, xor, "
                        f"checkerboard, stripes, perpendicular, x-mark, stars, "
                        f"triangle, square, pentagon, hexagon, heptagon, octagon, "
                        f"concentric_polygons, diamond, cross, s-curve, "
                        f"two_diagonals, ring, grid, swiss_roll_2d")
    
    return X, y


# ============== Helper Functions (Tight, Obvious Shapes) ==============

def make_spiral(n_samples, noise=0.0, random_state=42, n_classes=2, **kwargs):
    """Generate tight spiral dataset."""
    rng = np.random.RandomState(random_state)
    n_samples_per_class = n_samples // n_classes
    
    X = []
    y = []
    
    for class_id in range(n_classes):
        theta = np.linspace(class_id * 2 * np.pi / n_classes, 
                          (class_id + 1) * 2 * np.pi / n_classes,
                          n_samples_per_class)
        r = np.linspace(0.2, 1.5, n_samples_per_class)  # Tighter radius
        
        # Add spiral arms with tighter winding
        x = r * np.cos(theta + 3 * np.pi * r)  # More windings
        y_ = r * np.sin(theta + 3 * np.pi * r)
        
        X.append(np.column_stack([x, y_]))
        y.append(np.full(n_samples_per_class, class_id))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_xor(n_samples, noise=0.0, random_state=42):
    """Generate XOR dataset with tight, obvious separation."""
    rng = np.random.RandomState(random_state)
    
    # Generate points in tight [-1.5, 1.5] range
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # XOR labels with clear boundary
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_checkerboard(n_samples, noise=0.0, random_state=42, grid_size=3, **kwargs):
    """Generate tight checkerboard pattern."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Scale to grid coordinates
    grid_coords = (X + 1.5) / 3 * grid_size
    grid_coords = np.floor(grid_coords).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # Checkerboard label
    y = (grid_coords[:, 0] + grid_coords[:, 1]) % 2
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_stripes(n_samples, noise=0.0, random_state=42, n_stripes=4, **kwargs):
    """Generate tight vertical stripe pattern."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Label based on x-coordinate stripes
    scaled = (X[:, 0] + 1.5) / 3 * n_stripes
    y = np.floor(scaled).astype(int) % 2
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_perpendicular(n_samples, noise=0.0, random_state=42):
    """Generate tight perpendicular lines dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Tight perpendicular lines
    line_width = 0.2
    y = ((X[:, 0] > 0) & (np.abs(X[:, 1]) < line_width)).astype(int)
    y = np.logical_or(y, ((X[:, 1] > 0) & (np.abs(X[:, 0]) < line_width))).astype(int)
    y = 1 - y  # Invert to make two classes
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_x_mark(n_samples, noise=0.0, random_state=42):
    """Generate tight X-shaped dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Tight X: points near diagonals
    dist1 = np.abs(X[:, 0] - X[:, 1])
    dist2 = np.abs(X[:, 0] + X[:, 1])
    y = np.minimum(dist1, dist2) < 0.25
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y.astype(int)


def make_stars(n_samples, noise=0.0, random_state=42, n_points=5, **kwargs):
    """Generate tight star-like pattern."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Create tight star pattern
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(X[:, 1], X[:, 0])
    
    # Star shape with tighter radius
    star_mask = r < 0.8 + 0.4 * np.cos(n_points * theta)
    y = star_mask.astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_concentric_polygons(n_samples, n_sides, noise=0.0, 
                            random_state=42, n_rings=2, **kwargs):
    """Generate tight concentric polygons."""
    rng = np.random.RandomState(random_state)
    
    # Generate points in polar coordinates with tighter range
    r = rng.uniform(0, 1.5, n_samples)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    
    # Compute polygon radius with tighter bounds
    polygon_r = 1.0 / np.cos(np.pi / n_sides) * \
                np.cos(np.pi / n_sides) / np.cos(theta % (2 * np.pi / n_sides) - np.pi / n_sides)
    polygon_r = np.clip(polygon_r, 0.3, 1.0)
    
    # Label based on ring number
    ring_borders = np.linspace(0, 1.5, n_rings + 1)
    y = np.digitize(r, ring_borders[1:-1])
    
    # Convert to Cartesian with tighter scaling
    X = np.column_stack([
        r * np.cos(theta) * polygon_r * 0.8,
        r * np.sin(theta) * polygon_r * 0.8
    ])
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_diamond(n_samples, noise=0.0, random_state=42):
    """Generate tight diamond-shaped dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Tight diamond: |x| + |y| < 0.8
    y = (np.abs(X[:, 0]) + np.abs(X[:, 1]) < 0.8).astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_cross(n_samples, noise=0.0, random_state=42):
    """Generate tight cross-shaped dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Tight cross
    cross = ((np.abs(X[:, 0]) < 0.2) & (np.abs(X[:, 1]) < 1.0)) | \
            ((np.abs(X[:, 1]) < 0.2) & (np.abs(X[:, 0]) < 1.0))
    y = cross.astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_s_curve(n_samples, noise=0.0, random_state=42):
    """Generate tight S-curve shaped dataset."""
    rng = np.random.RandomState(random_state)
    
    t = rng.uniform(0, 2 * np.pi, n_samples)
    X = np.column_stack([
        1.5 * np.sin(t),
        t / np.pi - 1
    ])
    
    # Label based on y position with clear separation
    y = (X[:, 1] > 0).astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_two_diagonals(n_samples, noise=0.0, random_state=42):
    """Generate tight two diagonal lines dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Tight diagonals
    dist1 = np.abs(X[:, 0] - X[:, 1])
    dist2 = np.abs(X[:, 0] + X[:, 1])
    y = np.minimum(dist1, dist2) < 0.3
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y.astype(int)


def make_ring(n_samples, noise=0.0, random_state=42, inner_radius=0.3, outer_radius=1.0):
    """Generate tight ring (annulus) dataset."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    # Tight ring
    y = ((r > inner_radius) & (r < outer_radius)).astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_grid(n_samples, noise=0.0, random_state=42, grid_size=4, **kwargs):
    """Generate tight grid pattern."""
    rng = np.random.RandomState(random_state)
    
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    # Map to tight grid
    grid_coords = ((X + 1.5) / 3 * grid_size).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    # Label based on grid position parity
    y = (grid_coords[:, 0] + grid_coords[:, 1]) % 2
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_swiss_roll_2d(n_samples, noise=0.0, random_state=42):
    """Generate tight 2D projection of Swiss roll."""
    rng = np.random.RandomState(random_state)
    
    t = rng.uniform(1.5, 2.5 * np.pi, n_samples)  # Tighter range
    X = np.column_stack([
        t * np.cos(t) / 4,
        t * np.sin(t) / 4
    ])
    
    # Label based on position along the roll
    y = (t > 2.0 * np.pi).astype(int)
    
    # Add noise if requested
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


# ============== Example Usage ==============

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test all datasets with NO noise - tight, obvious shapes
    datasets = ['moons', 'blobs', 'spiral', 'xor', 'checkerboard', 
                'stripes', 'perpendicular', 'x-mark', 'stars', 'hexagon',
                'diamond', 'cross', 's-curve', 'two_diagonals', 'ring',
                'grid', 'swiss_roll_2d', 'concentric_polygons']
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    axes = axes.ravel()
    
    for idx, name in enumerate(datasets):
        if idx >= len(axes):
            break
        X, y = load_clf_2d(name, n_samples=500, noise=0.0)
        
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', 
                         s=15, alpha=0.8, edgecolors='none')
        axes[idx].set_title(name, fontsize=10)
        axes[idx].axis('equal')
        axes[idx].grid(True, alpha=0.2)
        axes[idx].set_xlim(-2, 2)
        axes[idx].set_ylim(-2, 2)
    
    # Hide empty subplots
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Example: Compare moons with and without noise
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    X_clean, y_clean = load_clf_2d('moons', n_samples=300, noise=0.0)
    X_noisy, y_noisy = load_clf_2d('moons', n_samples=300, noise=0.1)
    
    ax1.scatter(X_clean[:, 0], X_clean[:, 1], c=y_clean, cmap='coolwarm', s=20)
    ax1.set_title('Clean Moons (Tight)')
    ax1.axis('equal')
    
    ax2.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y_noisy, cmap='coolwarm', s=20)
    ax2.set_title('Noisy Moons')
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()