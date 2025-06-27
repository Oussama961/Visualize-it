import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

def configure_seaborn():
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")

def plot_linearity_test(f, title="Linearity Check"):
    """Visualize if f preserves linear combinations."""
    configure_seaborn()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    # Test 1: Additivity (f(x + y) = f(x) + f(y))
    x = np.array([1, 2])
    y = np.array([3, 1])
    lhs_add = f(x + y)
    rhs_add = f(x) + f(y)
    ax[0].quiver(0, 0, *x, color='blue', scale=10, label='x')
    ax[0].quiver(0, 0, *y, color='green', scale=10, label='y')
    ax[0].quiver(0, 0, *(x+y), color='purple', scale=10, label='x + y')
    ax[0].quiver(0, 0, *lhs_add, color='red', linestyle='--', scale=10, label='f(x + y)')
    ax[0].quiver(0, 0, *rhs_add, color='black', linestyle=':', scale=10, label='f(x) + f(y)')
    ax[0].set_title("Additivity Test")
    ax[0].legend()
    alpha = 2
    x = np.array([1, 1])
    lhs_homo = f(alpha * x)
    rhs_homo = alpha * f(x)
    ax[1].quiver(0, 0, *x, color='blue', scale=10, label='x')
    ax[1].quiver(0, 0, *(alpha*x), color='green', scale=10, label=f'{alpha}x')
    ax[1].quiver(0, 0, *lhs_homo, color='red', linestyle='--', scale=10, label=f'f({alpha}x)')
    ax[1].quiver(0, 0, *rhs_homo, color='black', linestyle=':', scale=10, label=f'{alpha}f(x)')
    ax[1].set_title("Homogeneity Test")
    ax[1].legend()    
    plt.suptitle(title)
    plt.show()


def linear_f(x):
    return np.dot([[2, 0], [0, 1]], x)

@interact
def interactive_linearity_test():
    plot_linearity_test(linear_f, "Linear Map: f(x) = [[2,0],[0,1]]x")