import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    
    # Write code here
    if type(x) is list:
        x = np.asarray(x, dtype=float)
        
    return 1 / (1 + np.exp(-x))