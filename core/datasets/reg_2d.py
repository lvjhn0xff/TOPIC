import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
import warnings

def load_reg_2d(name, n_samples=1000, noise=0.0, random_state=42, **kwargs):
    np.random.seed(random_state)
    rng = np.random.RandomState(random_state)
    
    # ========== CLASSIFICATION DATASETS ==========
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        
    elif name == 'blobs':
        n_clusters = kwargs.get('n_clusters', 2)
        X, y = make_blobs(n_samples=n_samples, n_features=2, 
                         centers=n_clusters, cluster_std=0.3,
                         random_state=random_state)
        
    elif name == 'circles':
        factor = kwargs.get('factor', 0.3)
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
    
    # ========== REGRESSION DATASETS ==========
    elif name == 'linear':
        X, y = make_linear(n_samples, noise, random_state, **kwargs)
        
    elif name == 'quadratic':
        X, y = make_quadratic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'cubic':
        X, y = make_cubic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'polynomial':
        degree = kwargs.get('degree', 3)
        X, y = make_polynomial(n_samples, degree, noise, random_state, **kwargs)
        
    elif name == 'sine':
        X, y = make_sine(n_samples, noise, random_state, **kwargs)
        
    elif name == 'cosine':
        X, y = make_cosine(n_samples, noise, random_state, **kwargs)
        
    elif name == 'trigonometric':
        X, y = make_trigonometric(n_samples, noise, random_state, **kwargs)
        
    elif name == 'exponential':
        X, y = make_exponential(n_samples, noise, random_state, **kwargs)
        
    elif name == 'logistic':
        X, y = make_logistic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'logarithmic':
        X, y = make_logarithmic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'gaussian':
        X, y = make_gaussian(n_samples, noise, random_state, **kwargs)
        
    elif name == 'sinc':
        X, y = make_sinc(n_samples, noise, random_state, **kwargs)
        
    elif name == 'hyperbolic':
        X, y = make_hyperbolic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'step':
        X, y = make_step(n_samples, noise, random_state, **kwargs)
        
    elif name == 'piecewise':
        X, y = make_piecewise(n_samples, noise, random_state, **kwargs)
        
    elif name == 'periodic':
        X, y = make_periodic(n_samples, noise, random_state, **kwargs)
        
    elif name == 'multi_modal':
        X, y = make_multi_modal(n_samples, noise, random_state, **kwargs)
        
    elif name == 'radial':
        X, y = make_radial(n_samples, noise, random_state, **kwargs)
        
    elif name == 'ripple':
        X, y = make_ripple(n_samples, noise, random_state, **kwargs)
        
    elif name == 'wave':
        X, y = make_wave(n_samples, noise, random_state, **kwargs)
        
    elif name == 'peak':
        X, y = make_peak(n_samples, noise, random_state, **kwargs)
        
    elif name == 'valley':
        X, y = make_valley(n_samples, noise, random_state, **kwargs)
        
    elif name == 'saddle':
        X, y = make_saddle(n_samples, noise, random_state, **kwargs)
        
    elif name == 'mountain':
        X, y = make_mountain(n_samples, noise, random_state, **kwargs)
        
    elif name == 'crater':
        X, y = make_crater(n_samples, noise, random_state, **kwargs)
        
    elif name == 'dune':
        X, y = make_dune(n_samples, noise, random_state, **kwargs)
        
    elif name == 'complex_polynomial':
        X, y = make_complex_polynomial(n_samples, noise, random_state, **kwargs)
        
    elif name == 'nonlinear':
        X, y = make_nonlinear(n_samples, noise, random_state, **kwargs)
        
    elif name == 'oscillating':
        X, y = make_oscillating(n_samples, noise, random_state, **kwargs)
        
    else:
        raise ValueError(f"Unknown dataset name: {name}. "
                        f"Available classification and regression datasets.")
    
    return X, y


# ========== CLASSIFICATION HELPER FUNCTIONS ==========

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
        r = np.linspace(0.2, 1.5, n_samples_per_class)
        
        x = r * np.cos(theta + 3 * np.pi * r)
        y_ = r * np.sin(theta + 3 * np.pi * r)
        
        X.append(np.column_stack([x, y_]))
        y.append(np.full(n_samples_per_class, class_id))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_xor(n_samples, noise=0.0, random_state=42):
    """Generate XOR dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_checkerboard(n_samples, noise=0.0, random_state=42, grid_size=3, **kwargs):
    """Generate tight checkerboard pattern."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    grid_coords = (X + 1.5) / 3 * grid_size
    grid_coords = np.floor(grid_coords).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    
    y = (grid_coords[:, 0] + grid_coords[:, 1]) % 2
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_stripes(n_samples, noise=0.0, random_state=42, n_stripes=4, **kwargs):
    """Generate tight vertical stripe pattern."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    scaled = (X[:, 0] + 1.5) / 3 * n_stripes
    y = np.floor(scaled).astype(int) % 2
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_perpendicular(n_samples, noise=0.0, random_state=42):
    """Generate tight perpendicular lines dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    line_width = 0.2
    y = ((X[:, 0] > 0) & (np.abs(X[:, 1]) < line_width)).astype(int)
    y = np.logical_or(y, ((X[:, 1] > 0) & (np.abs(X[:, 0]) < line_width))).astype(int)
    y = 1 - y
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_x_mark(n_samples, noise=0.0, random_state=42):
    """Generate tight X-shaped dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    dist1 = np.abs(X[:, 0] - X[:, 1])
    dist2 = np.abs(X[:, 0] + X[:, 1])
    y = np.minimum(dist1, dist2) < 0.25
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y.astype(int)


def make_stars(n_samples, noise=0.0, random_state=42, n_points=5, **kwargs):
    """Generate tight star-like pattern."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(X[:, 1], X[:, 0])
    
    star_mask = r < 0.8 + 0.4 * np.cos(n_points * theta)
    y = star_mask.astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_concentric_polygons(n_samples, n_sides, noise=0.0, 
                            random_state=42, n_rings=2, **kwargs):
    """Generate tight concentric polygons."""
    rng = np.random.RandomState(random_state)
    
    r = rng.uniform(0, 1.5, n_samples)
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    
    polygon_r = 1.0 / np.cos(np.pi / n_sides) * \
                np.cos(np.pi / n_sides) / np.cos(theta % (2 * np.pi / n_sides) - np.pi / n_sides)
    polygon_r = np.clip(polygon_r, 0.3, 1.0)
    
    ring_borders = np.linspace(0, 1.5, n_rings + 1)
    y = np.digitize(r, ring_borders[1:-1])
    
    X = np.column_stack([
        r * np.cos(theta) * polygon_r * 0.8,
        r * np.sin(theta) * polygon_r * 0.8
    ])
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_diamond(n_samples, noise=0.0, random_state=42):
    """Generate tight diamond-shaped dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = (np.abs(X[:, 0]) + np.abs(X[:, 1]) < 0.8).astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_cross(n_samples, noise=0.0, random_state=42):
    """Generate tight cross-shaped dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    cross = ((np.abs(X[:, 0]) < 0.2) & (np.abs(X[:, 1]) < 1.0)) | \
            ((np.abs(X[:, 1]) < 0.2) & (np.abs(X[:, 0]) < 1.0))
    y = cross.astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_s_curve(n_samples, noise=0.0, random_state=42):
    """Generate tight S-curve shaped dataset."""
    rng = np.random.RandomState(random_state)
    t = rng.uniform(0, 2 * np.pi, n_samples)
    X = np.column_stack([1.5 * np.sin(t), t / np.pi - 1])
    y = (X[:, 1] > 0).astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_two_diagonals(n_samples, noise=0.0, random_state=42):
    """Generate tight two diagonal lines dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    dist1 = np.abs(X[:, 0] - X[:, 1])
    dist2 = np.abs(X[:, 0] + X[:, 1])
    y = np.minimum(dist1, dist2) < 0.3
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y.astype(int)


def make_ring(n_samples, noise=0.0, random_state=42, inner_radius=0.3, outer_radius=1.0):
    """Generate tight ring dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = ((r > inner_radius) & (r < outer_radius)).astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_grid(n_samples, noise=0.0, random_state=42, grid_size=4, **kwargs):
    """Generate tight grid pattern."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    
    grid_coords = ((X + 1.5) / 3 * grid_size).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    y = (grid_coords[:, 0] + grid_coords[:, 1]) % 2
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


def make_swiss_roll_2d(n_samples, noise=0.0, random_state=42):
    """Generate tight 2D projection of Swiss roll."""
    rng = np.random.RandomState(random_state)
    t = rng.uniform(1.5, 2.5 * np.pi, n_samples)
    X = np.column_stack([t * np.cos(t) / 4, t * np.sin(t) / 4])
    y = (t > 2.0 * np.pi).astype(int)
    
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    
    return X, y


# ========== REGRESSION HELPER FUNCTIONS ==========

def make_linear(n_samples, noise=0.0, random_state=42, slope=1.0, intercept=0.0, **kwargs):
    """Generate linear regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    y = slope * X.flatten() + intercept
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_quadratic(n_samples, noise=0.0, random_state=42, a=1.0, b=0.0, c=0.0, **kwargs):
    """Generate quadratic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    y = a * X.flatten()**2 + b * X.flatten() + c
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_cubic(n_samples, noise=0.0, random_state=42, a=1.0, b=0.0, c=0.0, d=0.0, **kwargs):
    """Generate cubic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    y = a * X.flatten()**3 + b * X.flatten()**2 + c * X.flatten() + d
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_polynomial(n_samples, degree=3, noise=0.0, random_state=42, **kwargs):
    """Generate polynomial regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    
    # Random coefficients
    coeffs = rng.uniform(-1, 1, degree + 1)
    y = np.zeros(n_samples)
    for i, coeff in enumerate(coeffs):
        y += coeff * X.flatten()**i
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_sine(n_samples, noise=0.0, random_state=42, freq=2.0, amp=1.0, phase=0.0, **kwargs):
    """Generate sine wave regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, (n_samples, 1))
    y = amp * np.sin(freq * X.flatten() + phase)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_cosine(n_samples, noise=0.0, random_state=42, freq=2.0, amp=1.0, phase=0.0, **kwargs):
    """Generate cosine wave regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, (n_samples, 1))
    y = amp * np.cos(freq * X.flatten() + phase)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_trigonometric(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate trigonometric regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-np.pi, np.pi, (n_samples, 1))
    y = np.sin(X.flatten()) + np.cos(2 * X.flatten())
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_exponential(n_samples, noise=0.0, random_state=42, a=1.0, b=0.5, **kwargs):
    """Generate exponential regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0, 2, (n_samples, 1))
    y = a * np.exp(b * X.flatten())
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_logistic(n_samples, noise=0.0, random_state=42, a=1.0, b=0.0, c=1.0, **kwargs):
    """Generate logistic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, (n_samples, 1))
    y = a / (1 + np.exp(-b * (X.flatten() - c)))
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_logarithmic(n_samples, noise=0.0, random_state=42, a=1.0, b=0.0, **kwargs):
    """Generate logarithmic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0.1, 2, (n_samples, 1))
    y = a * np.log(X.flatten()) + b
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_gaussian(n_samples, noise=0.0, random_state=42, mu=0.0, sigma=0.5, amp=1.0, **kwargs):
    """Generate Gaussian (bell curve) regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-2, 2, (n_samples, 1))
    y = amp * np.exp(-(X.flatten() - mu)**2 / (2 * sigma**2))
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_sinc(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate sinc function regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-5, 5, (n_samples, 1))
    x = X.flatten()
    y = np.sin(x) / x
    y[x == 0] = 1.0  # Handle division by zero
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([x, rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_hyperbolic(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate hyperbolic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0.1, 2, (n_samples, 1))
    y = 1.0 / X.flatten()
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_step(n_samples, noise=0.0, random_state=42, threshold=0.0, **kwargs):
    """Generate step function regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    y = np.where(X.flatten() > threshold, 1.0, 0.0)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_piecewise(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate piecewise regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 1))
    x = X.flatten()
    y = np.where(x < -0.5, -x - 0.5, np.where(x < 0.5, x**2, x - 0.5))
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([x, rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_periodic(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate periodic regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-2*np.pi, 2*np.pi, (n_samples, 1))
    y = np.sin(X.flatten()) * np.cos(2 * X.flatten())
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_multi_modal(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate multi-modal regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-3, 3, (n_samples, 1))
    y = 2 * np.exp(-(X.flatten() + 1)**2) + 1.5 * np.exp(-(X.flatten() - 1)**2)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    X = np.column_stack([X.flatten(), rng.uniform(-0.1, 0.1, n_samples)])
    return X, y


def make_radial(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate radial 2D regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = np.exp(-r**2)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_ripple(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate ripple 2D regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = np.sin(5 * r) * np.exp(-r)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_wave(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D wave regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.sin(X[:, 0] * 3) * np.cos(X[:, 1] * 2)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_peak(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D peak regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.exp(-(X[:, 0]**2 + X[:, 1]**2) / 0.25)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_valley(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D valley regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = (X[:, 0]**2 + X[:, 1]**2) / 2
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_saddle(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D saddle regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = X[:, 0]**2 - X[:, 1]**2
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_mountain(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D mountain regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = (1 - X[:, 0])**2 + 100 * (X[:, 1] - X[:, 0]**2)**2  # Rosenbrock-like
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_crater(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D crater regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = -np.exp(-r**2 / 0.1) + 0.5 * np.exp(-r**2 / 0.5)
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_dune(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate 2D dune regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.sin(X[:, 0] * 2) * np.exp(-np.abs(X[:, 1]))
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_complex_polynomial(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate complex polynomial 2D regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = X[:, 0]**3 - X[:, 1]**2 + 0.5 * X[:, 0] * X[:, 1] + X[:, 0]**2 * X[:, 1]
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_nonlinear(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate nonlinear 2D regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.sin(X[:, 0] * X[:, 1] * 3) + 0.5 * X[:, 0]**2 * X[:, 1]
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


def make_oscillating(n_samples, noise=0.0, random_state=42, **kwargs):
    """Generate oscillating 2D regression dataset."""
    rng = np.random.RandomState(random_state)
    X = rng.uniform(-1.5, 1.5, (n_samples, 2))
    y = np.sin(X[:, 0] * 4) * np.cos(X[:, 1] * 3) * np.exp(-0.5 * (X[:, 0]**2 + X[:, 1]**2))
    
    if noise > 0:
        y += rng.normal(0, noise, n_samples)
    
    return X, y


# ========== Example Usage ==========

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # ===== Classification Examples =====
    print("Classification Datasets:")
    class_datasets = ['moons', 'spiral', 'xor', 'checkerboard', 'hexagon', 'ring']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, name in enumerate(class_datasets):
        X, y = load_reg_2d(name, n_samples=300, noise=0.0)
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20, alpha=0.8)
        axes[idx].set_title(f'Classification: {name}')
        axes[idx].axis('equal')
        axes[idx].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    # ===== Regression Examples =====
    print("Regression Datasets:")
    
    # 1D regression examples
    reg_1d = ['linear', 'quadratic', 'sine', 'logistic', 'step', 'multi_modal']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, name in enumerate(reg_1d):
        X, y = load_reg_2d(name, n_samples=300, noise=0.05)
        axes[idx].scatter(X[:, 0], y, s=10, alpha=0.6)
        axes[idx].set_title(f'Regression: {name}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    # 2D regression examples (surface plots)
    reg_2d = ['radial', 'ripple', 'wave', 'peak', 'saddle', 'crater']
    
    fig = plt.figure(figsize=(18, 10))
    
    for idx, name in enumerate(reg_2d):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        X, y = load_reg_2d(name, n_samples=1000, noise=0.0)
        
        # Create surface plot
        x = X[:, 0]
        z = X[:, 1]
        y_vals = y
        
        surf = ax.scatter(x, z, y_vals, c=y_vals, cmap='viridis', 
                         s=10, alpha=0.6, marker='o')
        ax.set_title(f'2D Regression: {name}')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # ===== Compare with Noise =====
    print("\nComparing clean vs noisy regression:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    X_clean, y_clean = load_reg_2d('sine', n_samples=200, noise=0.0)
    X_noisy, y_noisy = load_reg_2d('sine', n_samples=200, noise=0.3)
    
    ax1.scatter(X_clean[:, 0], y_clean, s=15, alpha=0.6)
    ax1.set_title('Clean Sine Wave')
    ax1.grid(True, alpha=0.2)
    
    ax2.scatter(X_noisy[:, 0], y_noisy, s=15, alpha=0.6)
    ax2.set_title('Noisy Sine Wave')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()