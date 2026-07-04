import numpy as np 

def percentiles(x): 
    return [
        np.percentile(x, 1),
        np.percentile(x, 5),
        np.percentile(x, 10),
        np.percentile(x, 20),
        np.percentile(x, 25),
        np.percentile(x, 50),
        np.percentile(x, 75),
        np.percentile(x, 80),
        np.percentile(x, 90),
        np.percentile(x, 95),
        np.percentile(x, 99)
    ]