import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_diagonal(n, diag_values):
    """Create a diagonal matrix with given diagonal entries."""
    return np.diag(diag_values)

def create_triangular(n, upper=True):
    """Create upper/lower triangular matrix with random entries."""
    matrix = np.random.randint(-5, 5, (n, n))
    return np.triu(matrix) if upper else np.tril(matrix)

def create_scalar(n, scalar):
    """Create a scalar matrix (scalar * identity)."""
    return scalar * np.eye(n)

def create_column_vector(n):
    """Create a column vector (n x 1 matrix)."""
    return np.random.randint(-5, 5, (n, 1))

def create_row_vector(m):
    """Create a row vector (1 x m matrix)."""
    return np.random.randint(-5, 5, (1, m))


def configure_seaborn():
    sns.set_theme(context="notebook", style="whitegrid")

def plot_matrix(matrix, title="Matrix", annotate=True):
    """Plot a matrix using a heatmap."""
    configure_seaborn()
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        matrix, annot=annotate, fmt="d", cmap="coolwarm",
        cbar=False, linewidths=0.5, linecolor="black"
    )
    ax.set_title(title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


def plot_vector(vector, title="Vector"):
    """Plot a column or row vector as a bar chart."""
    configure_seaborn()
    plt.figure(figsize=(4, 4))
    plt.bar(range(len(vector)), vector.flatten())
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

