import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from ipywidgets import interact

from mpl_toolkits.mplot3d import Axes3D


def configure_seaborn():
    """Configure Seaborn styling for all plots."""
    sns.set_theme(context="notebook", style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (8, 6)

def plot_vector_addition(u, v, title="Vector Addition"):
    """Plot vectors u, v, and their sum u + v with Seaborn styling."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Plot vectors
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, 
              color=sns.color_palette()[0], label=f'u = {u}', width=0.02)
    ax.quiver(u[0], u[1], v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color=sns.color_palette()[1], label=f'v = {v}', width=0.02)
    ax.quiver(0, 0, u[0]+v[0], u[1]+v[1], angles='xy', scale_units='xy', scale=1, 
              color=sns.color_palette()[2], label='u + v', width=0.02)
    
    # Styling with Seaborn
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x-axis', fontsize=12)
    ax.set_ylabel('y-axis', fontsize=12)
    sns.despine(left=True, bottom=True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("../Outputs/Algebra/vector_addition.png", dpi=300)
    return fig  # Return figure for reuse in other contexts

def plot_scalar_multiplication(alpha, v, title="Scalar Multiplication"):
    """Plot scalar multiplication αv with Seaborn styling."""
    configure_seaborn()
    scaled_v = (alpha * v[0], alpha * v[1])
    fig, ax = plt.subplots()
    
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color=sns.color_palette()[0], label=f'v = {v}', width=0.02)
    ax.quiver(0, 0, scaled_v[0], scaled_v[1], angles='xy', scale_units='xy', scale=1, 
              color=sns.color_palette()[3], label=f'{alpha}v', width=0.02)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x-axis', fontsize=12)
    ax.set_ylabel('y-axis', fontsize=12)
    sns.despine(left=True, bottom=True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("../Outputs/Algebra/Scaler Multiplication.png", dpi=300)
    return fig


# vector spaces _ properties plots ___________________________________________________________________________________________
def plot_property_i(a, x, title="Property (i): ax = 0 iff a=0 or x=0"):
    """Visualize ax = 0 when either a=0 or x=0."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Compute ax
    result = (a * x[0], a * x[1])
    color = 'red' if a == 0 or x == (0, 0) else 'gray'
    
    # Plot vector
    ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1,
              color=color, width=0.02, label=f'{a} * {x} = {result}')
    
    # Highlight zero vector
    if result == (0, 0):
        ax.scatter(0, 0, color='black', s=100, zorder=5, marker='x', label='Zero Vector')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_property_ii(a, x, y, title="Property (ii): a(x - y) = ax - ay"):
    """Visualize scalar multiplication over vector subtraction."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Compute both sides of the equation
    lhs = (a * (x[0] - y[0]), a * (x[1] - y[1]))  # a(x - y)
    rhs = (a*x[0] - a*y[0], a*x[1] - a*y[1])      # ax - ay
    
    # Plot LHS (a(x - y))
    ax.quiver(0, 0, lhs[0], lhs[1], color='blue', width=0.02, 
              label=f'a(x - y) = {lhs}', scale=1)
    # Plot RHS (ax - ay)
    ax.quiver(0, 0, rhs[0], rhs[1], color='red', linestyle='--', width=0.02,
              label=f'ax - ay = {rhs}', scale=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig


# vector spaces _ properties plots ___________________________________________________________________________________________
def plot_property_i(a, x, title="Property (i): ax = 0 iff a=0 or x=0"):
    """Visualize ax = 0 when either a=0 or x=0."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Compute ax
    result = (a * x[0], a * x[1])
    color = 'red' if a == 0 or x == (0, 0) else 'gray'
    
    # Plot vector
    ax.quiver(0, 0, result[0], result[1], angles='xy', scale_units='xy', scale=1,
              color=color, width=0.02, label=f'{a} * {x} = {result}')
    
    # Highlight zero vector
    if result == (0, 0):
        ax.scatter(0, 0, color='black', s=100, zorder=5, marker='x', label='Zero Vector')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_property_ii(a, x, y, title="Property (ii): a(x - y) = ax - ay"):
    """Visualize scalar multiplication over vector subtraction."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Compute both sides of the equation
    lhs = (a * (x[0] - y[0]), a * (x[1] - y[1]))  # a(x - y)
    rhs = (a*x[0] - a*y[0], a*x[1] - a*y[1])      # ax - ay
    
    # Plot LHS (a(x - y))
    ax.quiver(0, 0, lhs[0], lhs[1], color='blue', width=0.02, 
              label=f'a(x - y) = {lhs}', scale=1)
    # Plot RHS (ax - ay)
    ax.quiver(0, 0, rhs[0], rhs[1], color='red', linestyle='--', width=0.02,
              label=f'ax - ay = {rhs}', scale=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

# vector spaces _ linear combinations ___________________________________________________________________________________________
def plot_linear_combination(vectors, coefficients, title="Linear Combination"):
    """Plot a linear combination of vectors."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Compute the linear combination
    result = np.array([0.0, 0.0])
    for (x, y), coeff in zip(vectors, coefficients):
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, 
                  color=sns.color_palette(), width=0.02, alpha=0.5,
                  label=f'{coeff} * ({x}, {y})')
        result += coeff * np.array([x, y])
    
    # Plot the resultant vector
    ax.quiver(0, 0, result[0], result[1], color='black', width=0.03,
              label=f'Result: {tuple(result.round(2))}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig


# vector spaces _ product spaces ___________________________________________________________________________________________
def plot_vector_addition_product(v1, v2, title="Product Space Addition: (x1+y1, x2+y2)"):
    """Visualize component-wise addition in E1 × E2."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Extract components (assuming E1 and E2 are 1D for simplicity)
    x1, x2 = v1
    y1, y2 = v2
    result = (x1 + y1, x2 + y2)
    
    # Plot vectors from E1 (x-axis) and E2 (y-axis)
    ax.quiver(0, 0, x1, 0, color='blue', width=0.02, scale=1, label=f'E1: x1 = {x1}')
    ax.quiver(0, 0, 0, x2, color='green', width=0.02, scale=1, label=f'E2: x2 = {x2}')
    ax.quiver(0, 0, y1, 0, color='blue', linestyle='--', width=0.02)
    ax.quiver(0, 0, 0, y2, color='green', linestyle='--', width=0.02)
    ax.quiver(0, 0, result[0], result[1], color='red', width=0.03, 
              label=f'Sum: ({result[0]}, {result[1]})')
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('E1 (Axis 1)', fontsize=12)
    ax.set_ylabel('E2 (Axis 2)', fontsize=12)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_scalar_multiplication_product(alpha, v, title="Scalar Multiplication: (αx1, αx2)"):
    """Visualize component-wise scalar multiplication in E1 × E2."""
    configure_seaborn()
    scaled_v = (alpha * v[0], alpha * v[1])
    fig, ax = plt.subplots()
    
    # Original and scaled vectors
    ax.quiver(0, 0, v[0], 0, color='blue', width=0.02, label=f'E1: x1 = {v[0]}')
    ax.quiver(0, 0, 0, v[1], color='green', width=0.02, label=f'E2: x2 = {v[1]}')
    ax.quiver(0, 0, scaled_v[0], 0, color='blue', linestyle='--', width=0.02)
    ax.quiver(0, 0, 0, scaled_v[1], color='green', linestyle='--', width=0.02)
    ax.quiver(0, 0, scaled_v[0], scaled_v[1], color='red', width=0.03,
              label=f'Scaled: ({scaled_v[0]}, {scaled_v[1]})')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('E1 (Axis 1)', fontsize=12)
    ax.set_ylabel('E2 (Axis 2)', fontsize=12)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig



# vector spaces _ subspaces _________________________________________________________________________________________
def plot_subspace_example(vectors, title="Subspace Example"):
    """Plot vectors in a candidate subspace and their combinations."""
    configure_seaborn()
    fig, ax = plt.subplots()    
    # Highlight the subspace (e.g., a line y = 2x)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, 2*x, color='gray', linestyle='--', alpha=0.3, label='Candidate Subspace: y = 2x')    
    # Plot vectors and their linear combinations
    for v in vectors:
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color=sns.color_palette(), width=0.02, alpha=0.7)    
    # Check closure: plot a random combination
    coeffs = np.random.uniform(-1, 1, size=len(vectors))
    combination = np.sum([c * np.array(v) for c, v in zip(coeffs, vectors)], axis=0)
    ax.quiver(0, 0, combination[0], combination[1], color='black', width=0.03,
              label=f'Combination: {tuple(combination.round(2))}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_non_subspace_example(vectors, title="Non-Subspace Example"):
    """Plot a set that is NOT a subspace (e.g., missing zero vector)."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Highlight a non-subspace (e.g., line y = 2x + 1)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, 2*x + 1, color='red', linestyle='--', alpha=0.3, label='Non-Subspace: y = 2x + 1')
    
    # Plot vectors and their combinations
    for v in vectors:
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color=sns.color_palette(), width=0.02, alpha=0.7)
    
    # Check closure failure
    coeffs = np.random.uniform(-1, 1, size=len(vectors))
    combination = np.sum([c * np.array(v) for c, v in zip(coeffs, vectors)], axis=0)
    ax.quiver(0, 0, combination[0], combination[1], color='black', width=0.03,
              label=f'Combination: {tuple(combination.round(2))}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig


# vector spaces __ span visualizations _________________________________________________________________________________________
def plot_intersection_subspaces(subspace1, subspace2, title="Intersection of Subspaces"):
    """Visualize the intersection of two subspaces (e.g., lines in 2D)."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Plot subspaces (example: y = 2x and y = -x)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, subspace1(x), color='blue', linestyle='--', alpha=0.3, label='Subspace 1: y = 2x')
    ax.plot(x, subspace2(x), color='green', linestyle='--', alpha=0.3, label='Subspace 2: y = -x')
    # Highlight intersection (here, only the origin)
    ax.scatter(0, 0, color='red', s=100, zorder=5, label='Intersection: {0}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_union_not_subspace(subspace1, subspace2, title="Union of Subspaces ≠ Subspace"):
    """Show that the union of two subspaces fails closure."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Plot subspaces (e.g., x-axis and y-axis)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, subspace1(x), color='blue', linestyle='--', alpha=0.3, label='Subspace 1: x-axis')
    ax.plot(x, subspace2(x), color='green', linestyle='--', alpha=0.3, label='Subspace 2: y-axis')
    
    # Add vectors from each subspace and their sum (not in the union)
    ax.quiver(0, 0, 1, 0, color='blue', width=0.02, scale=1, label='(1, 0) ∈ Subspace 1')
    ax.quiver(0, 0, 0, 1, color='green', width=0.02, scale=1, label='(0, 1) ∈ Subspace 2')
    ax.quiver(0, 0, 1, 1, color='red', width=0.03, scale=1, label='(1, 1) ∉ Union')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

def plot_span_A(vectors, title="Span of a Subset Vect(A)"):
    """Visualize the span of vectors in A as the smallest containing subspace."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Plot all linear combinations of vectors in A
    for v in vectors:
        ax.quiver(0, 0, v[0], v[1], color='gray', width=0.02, alpha=0.5)
    
    # Generate and plot random linear combinations
    for _ in range(20):
        coeffs = np.random.uniform(-2, 2, size=len(vectors))
        combination = np.sum([c * np.array(v) for c, v in zip(coeffs, vectors)], axis=0)
        ax.scatter(combination[0], combination[1], color='blue', alpha=0.4, s=50)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    plt.tight_layout()
    return fig


# vector spaces __ finite span   _________________________________________________________________________________________
def plot_finite_span(vectors, num_combinations=50, title="Vect(A) = Span of Finite Set"):
    """Visualize the span of a finite set of vectors as all linear combinations."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Plot the original vectors
    colors = sns.color_palette("husl", len(vectors))
    for i, (x, y) in enumerate(vectors):
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
                  color=colors[i], width=0.02, alpha=0.8, label=f'$a_{i+1} = ({x}, {y})$')
    # Generate and plot random linear combinations
    for _ in range(num_combinations):
        coefficients = np.random.uniform(-2, 2, size=len(vectors))
        combination = np.sum([c * np.array(v) for c, v in zip(coefficients, vectors)], axis=0)
        ax.scatter(combination[0], combination[1], color='gray', alpha=0.4, s=30)
    # Highlight the subspace (e.g., line/plane spanned by vectors)
    if len(vectors) == 1:
        x = np.linspace(-5, 5, 100)
        y = (vectors[0][1]/vectors[0][0]) * x if vectors[0][0] != 0 else x*0
        ax.plot(x, y, linestyle='--', color='red', alpha=0.5, label='Span (1D)')
    elif len(vectors) == 2:
        # For 2 vectors, span is the plane unless they are colinear
        ax.plot([-5, 5], [-5, 5], linestyle='--', color='red', alpha=0.3, label='Span (2D)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

# vector spaces __ subspace sums _________________________________________________________________________________________
# 2D Visualization (Sum of Lines/Planes)
def plot_subspace_sum_2d(subspaces, title="Sum of Subspaces (F₁ + F₂)"):
    """Visualize the sum of two subspaces in 2D."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Plot each subspace (e.g., lines through the origin)
    colors = sns.color_palette("husl", len(subspaces))
    for i, (slope, intercept) in enumerate(subspaces):
        x = np.linspace(-5, 5, 100)
        y = slope * x + intercept
        ax.plot(x, y, linestyle='--', color=colors[i], alpha=0.5, 
                label=f'F_{i+1}: y = {slope}x')
    # Highlight vectors from each subspace and their sum
    for i, (slope, _) in enumerate(subspaces):
        v = np.array([1, slope])  # Direction vector of the subspace
        ax.quiver(0, 0, v[0], v[1], color=colors[i], scale=10, width=0.02)
    # Generate a vector in F₁ + F₂ (sum of random vectors from each subspace)
    v1 = np.array([1, subspaces[0][0]])
    v2 = np.array([1, subspaces[1][0]])
    sum_vector = v1 + v2
    ax.quiver(0, 0, sum_vector[0], sum_vector[1], color='black', width=0.03,
              label=f'v₁ + v₂ = {sum_vector}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

# 3D Visualization (Sum of Planes/Lines)
def plot_subspace_sum_3d(subspace1, subspace2, title="Sum of Subspaces in 3D"):
    """Visualize the sum of two subspaces in 3D (e.g., two planes)."""
    configure_seaborn()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Subspace 1: Generate points for a plane (e.g., z = 0)
    x1, y1 = np.meshgrid(np.linspace(-5, 5, 10), np.leshgrid(np.linspace(-5, 5, 10)))
    z1 = np.zeros_like(x1)
    ax.plot_surface(x1, y1, z1, alpha=0.3, color='blue', label='F₁: z=0')
    # Subspace 2: Generate points for another plane (e.g., x = 0)
    y2, z2 = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    x2 = np.zeros_like(y2)
    ax.plot_surface(x2, y2, z2, alpha=0.3, color='green', label='F₂: x=0')
    # Add vectors from each subspace and their sum
    v1 = np.array([2, 2, 0])  # In F₁
    v2 = np.array([0, 2, 2])  # In F₂
    sum_v = v1 + v2
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', lw=2)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='green', lw=2)
    ax.quiver(0, 0, 0, sum_v[0], sum_v[1], sum_v[2], color='red', lw=2,
              label=f'v₁ + v₂ = {sum_v}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.legend()
    plt.tight_layout()
    return fig

# vector spaces __ direct sums  _________________________________________________________________________________________
# 2D Direct Sum (Supplementary Subspaces)
def plot_direct_sum_2d(subspace1, subspace2, title="E = F ⊕ G (Supplementary Subspaces)"):
    """Visualize supplementary subspaces F and G in 2D."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Plot subspaces (e.g., x-axis and y-axis)
    x = np.linspace(-5, 5, 100)
    ax.plot(x, subspace1(x), color='blue', linestyle='--', alpha=0.3, label='F: x-axis')
    ax.plot(x, subspace2(x), color='green', linestyle='--', alpha=0.3, label='G: y-axis')
    # Plot vectors from F and G and their unique sum
    v_F = np.array([3, 0])  # Vector in F
    v_G = np.array([0, 2])  # Vector in G
    v_sum = v_F + v_G
    ax.quiver(0, 0, v_F[0], v_F[1], color='blue', scale=10, width=0.02, label='v ∈ F')
    ax.quiver(0, 0, v_G[0], v_G[1], color='green', scale=10, width=0.02, label='w ∈ G')
    ax.quiver(0, 0, v_sum[0], v_sum[1], color='red', width=0.03, label='v + w')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    sns.despine()
    ax.legend()
    plt.tight_layout()
    return fig

# 3D Direct Sum (Plane ⊕ Line)
def plot_direct_sum_3d(title="E = F ⊕ G (3D Supplementary Subspaces)"):
    """Visualize supplementary subspaces: a plane and a line in 3D."""
    configure_seaborn()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Subspace F: Plane z = 0
    x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    z = np.zeros_like(x)
    ax.plot_surface(x, y, z, alpha=0.3, color='blue', label='F: z=0')
    # Subspace G: Line along z-axis (x=0, y=0)
    z_line = np.linspace(-5, 5, 100)
    ax.plot(np.zeros_like(z_line), np.zeros_like(z_line), z_line, 
            color='green', linestyle='--', alpha=0.5, label='G: z-axis')
    # Unique decomposition example
    v_F = np.array([2, 3, 0])  # In F (plane)
    v_G = np.array([0, 0, 4])  # In G (line)
    v_sum = v_F + v_G
    ax.quiver(0, 0, 0, v_F[0], v_F[1], v_F[2], color='blue', lw=2)
    ax.quiver(0, 0, 0, v_G[0], v_G[1], v_G[2], color='green', lw=2)
    ax.quiver(0, 0, 0, v_sum[0], v_sum[1], v_sum[2], color='red', lw=2, label='v + w')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.legend()
    plt.tight_layout()
    return fig

# algebra_2_visualizations/
# ├── bases/
# │   ├── __init__.py
# │   ├── generating_families.py      # Spanning sets
# │   ├── linear_independence.py      # Linearly independent sets
# │   ├── bases_and_dimension.py      # Bases and dimension theorems
# │   └── incomplete_basis.py         # Theorem of incomplete basis
# └── master_notebook.ipynb           # Master notebook to render all visuals

# generating families 
# Spanning sets
def plot_generating_family_2d(v1, v2):
    """Show how vectors v1 and v2 span ℝ²."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot vectors
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.02, label=f'v1 = {v1}')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='green', width=0.02, label=f'v2 = {v2}')
    # Generate random linear combinations
    for _ in range(50):
        a, b = np.random.uniform(-2, 2, 2)
        combo = a * np.array(v1) + b * np.array(v2)
        ax.scatter(combo[0], combo[1], color='red', alpha=0.4)
    ax.set_title("Generating Family: Span of v1 and v2")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.legend()
    plt.show()

@interact(v1_x=(-2.0, 2.0), v1_y=(-2.0, 2.0), v2_x=(-2.0, 2.0), v2_y=(-2.0, 2.0))
def update_generating_family(v1_x=1, v1_y=0, v2_x=0, v2_y=1):
    plot_generating_family_2d((v1_x, v1_y), (v2_x, v2_y))

# linear independence
def plot_linear_independence(v1, v2):
    """Show if two vectors are linearly independent."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot vectors
    ax.quiver(0, 0, v1[0], v1[1], color='blue', width=0.02, label=f'v1 = {v1}')
    ax.quiver(0, 0, v2[0], v2[1], color='green', width=0.02, label=f'v2 = {v2}')
    # Check if dependent (collinear)
    is_dependent = np.isclose(np.linalg.det([v1, v2]), 0)
    ax.set_title(f"Linear Independence: {not is_dependent}")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.legend()
    plt.show()

@interact(v1_x=(-2.0, 2.0), v1_y=(-2.0, 2.0), v2_x=(-2.0, 2.0), v2_y=(-2.0, 2.0))
def update_linear_independence(v1_x=1, v1_y=2, v2_x=2, v2_y=4):
    plot_linear_independence((v1_x, v1_y), (v2_x, v2_y))

# bases and dimension
def plot_basis_3d(v1, v2, v3):
    """Show 3D basis vectors and their combinations."""
    configure_seaborn()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label=f'v1 = {v1}')
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='green', label=f'v2 = {v2}')
    ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='red', label=f'v3 = {v3}')
    # Check if basis (determinant ≠ 0)
    matrix = np.array([v1, v2, v3])
    is_basis = not np.isclose(np.linalg.det(matrix), 0)
    ax.set_title(f"Basis of ℝ³: {is_basis}")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.legend()
    plt.show()

@interact(v1_x=(-2.0, 2.0), v1_y=(-2.0, 2.0), v1_z=(-2.0, 2.0), v2_x=(-2.0, 2.0), v2_y=(-2.0, 2.0), v2_z=(-2.0, 2.0), v3_x=(-2.0, 2.0), v3_y=(-2.0, 2.0), v3_z=(-2.0, 2.0))
def update_basis(v1_x=1, v1_y=0, v1_z=0,v2_x=0, v2_y=1, v2_z=0,v3_x=0, v3_y=0, v3_z=1):
    plot_basis_3d((v1_x, v1_y, v1_z), (v2_x, v2_y, v2_z), (v3_x, v3_y, v3_z))

# incomplete basis
def plot_incomplete_basis():
    """Start with a vector and extend to a basis."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Initial independent vector
    v1 = np.array([2, 1])
    ax.quiver(0, 0, v1[0], v1[1], color='blue', width=0.02, label='Initial independent vector')
    # Add a second vector to form a basis
    v2 = np.array([-1, 2])
    ax.quiver(0, 0, v2[0], v2[1], color='green', width=0.02, label='Added to form basis')
    ax.set_title("Theorem of Incomplete Basis: Extend {v1} to {v1, v2}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    plt.show()



def plot_affine_space(points, vectors, title="Affine Space Visualization"):
    """Plot points and vectors in an affine space."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Plot points
    for (x, y), label in points.items():
        ax.scatter(x, y, s=100, label=f'Point {label}')
    # Plot vectors as arrows between points
    for (start, end), vec in vectors.items():
        dx = points[end][0] - points[start][0]
        dy = points[end][1] - points[start][1]
        ax.quiver(points[start][0], points[start][1], dx, dy, 
                  angles='xy', scale_units='xy', scale=1, 
                  color='red', width=0.005, label=f'Vector {vec}')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.show()

# Usage exemple:
points = {'A': (1, 1),'B': (3, 4),'C': (4, 6)}
vectors = {('A', 'B'): 'AB',('B', 'C'): 'BC',('A', 'C'): 'AC'}
plot_affine_space(points, vectors, title="Chasles Relation: AB + BC = AC")


# affine spaces __ affine frame 2D ____________________________________________________________________________________
def plot_affine_frame_2d(origin, e1, e2, M_coords):
    """Plot affine frame (O, e1, e2) and point M with coordinates."""
    configure_seaborn()
    fig, ax = plt.subplots()
    # Basis vectors from origin
    ax.quiver(*origin, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.005, label=f'e1 = {e1}')
    ax.quiver(*origin, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
              color='green', width=0.005, label=f'e2 = {e2}')
    # Point M in this frame: OM = x1*e1 + x2*e2
    x1, x2 = M_coords
    OM = x1 * np.array(e1) + x2 * np.array(e2)
    M = (origin[0] + OM[0], origin[1] + OM[1])
    ax.scatter(*M, s=100, color='red', label=f'M = ({x1}, {x2}) in frame')
    # Grid and styling
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_title("2D Affine Frame: Coordinates Depend on Basis")
    ax.legend()
    plt.show()

@interact(e1_x=(-2.0, 2.0, 0.5), e1_y=(-2.0, 2.0, 0.5),e2_x=(-2.0, 2.0, 0.5), e2_y=(-2.0, 2.0, 0.5),x1=(-5, 5, 1), x2=(-5, 5, 1))
def update_frame_2d(
    e1_x=1, e1_y=0,  # Default basis e1=(1,0)
    e2_x=0, e2_y=1,  # Default basis e2=(0,1)
    x1=1, x2=1       # Coordinates of M in this basis
):
    plot_affine_frame_2d(origin=(0, 0), e1=(e1_x, e1_y), e2=(e2_x, e2_y), 
                         M_coords=(x1, x2))


# affine spaces __ affine frame 3D  _______________________________________________________________________________________________
def plot_affine_frame_3d(origin, e1, e2, e3, M_coords):
    """Plot 3D affine frame (O, e1, e2, e3) and point M."""
    configure_seaborn()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Basis vectors
    ax.quiver(*origin, *e1, color='blue', label=f'e1 = {e1}')
    ax.quiver(*origin, *e2, color='green', label=f'e2 = {e2}')
    ax.quiver(*origin, *e3, color='purple', label=f'e3 = {e3}')
    # Point M: OM = x1*e1 + x2*e2 + x3*e3
    x1, x2, x3 = M_coords
    OM = x1*np.array(e1) + x2*np.array(e2) + x3*np.array(e3)
    M = (origin[0] + OM[0], origin[1] + OM[1], origin[2] + OM[2])
    ax.scatter(*M, s=100, color='red', label=f'M = ({x1}, {x2}, {x3})')    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("3D Affine Frame: Coordinates Depend on Basis")
    ax.legend()
    plt.show()

@interact(x1=(-5, 5, 1), x2=(-5, 5, 1), x3=(-5, 5, 1))
def update_frame_3d(x1=1, x2=1, x3=1):
    plot_affine_frame_3d(
        origin=(0, 0, 0),
        e1=(1, 0, 0),  # Fixed canonical basis for simplicity
        e2=(0, 1, 0),
        e3=(0, 0, 1),
        M_coords=(x1, x2, x3)
    )


# affine spaces __ affine subspaces and directions ___________________________________________________________________
def plot_affine_line(origin, direction_vector, title="Affine Line (Direction F)"):
    """Plot an affine line in 2D."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(8, 6))
    # Line equation: origin + t * direction_vector
    t = np.linspace(-5, 5, 100)
    x = origin[0] + t * direction_vector[0]
    y = origin[1] + t * direction_vector[1]
    ax.plot(x, y, linestyle='--', alpha=0.7, label=f'Line: {origin} + t*{direction_vector}')
    ax.scatter(origin[0], origin[1], color='red', s=100, label=f'Origin A = {origin}')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig("../Outputs/Algebra/affine_line.png", dpi=300)
    return fig

@interact(
    origin_x=(-5, 5), origin_y=(-5, 5),
    dir_x=(-2.0, 2.0, 0.5), dir_y=(-2.0, 2.0, 0.5)
)
def update_affine_line(origin_x=1, origin_y=1, dir_x=1, dir_y=2):
    plot_affine_line((origin_x, origin_y), (dir_x, dir_y))

def plot_intersection_lines(line1_origin, line1_dir, line2_origin, line2_dir, title="Intersection of Affine Lines"):
    """Plot two affine lines and their intersection (if any)."""
    configure_seaborn()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Line 1
    t = np.linspace(-5, 5, 100)
    x1 = line1_origin[0] + t * line1_dir[0]
    y1 = line1_origin[1] + t * line1_dir[1]
    ax.plot(x1, y1, label=f'Line 1: {line1_origin} + t*{line1_dir}', color='blue')
    
    # Plot Line 2
    x2 = line2_origin[0] + t * line2_dir[0]
    y2 = line2_origin[1] + t * line2_dir[1]
    ax.plot(x2, y2, label=f'Line 2: {line2_origin} + t*{line2_dir}', color='green')
    
    # Find intersection
    # Using parametric equation solution
    # For line 1: P1 + t1*v1 = line1_origin + t1*line1_dir
    # For line 2: P2 + t2*v2 = line2_origin + t2*line2_dir
    # At intersection: P1 + t1*v1 = P2 + t2*v2
    
    # Set up linear system: line1_origin + t1*line1_dir = line2_origin + t2*line2_dir
    # Rearranging: t1*line1_dir - t2*line2_dir = line2_origin - line1_origin
    
    A = np.column_stack((line1_dir, -np.array(line2_dir)))
    b = np.array(line2_origin) - np.array(line1_origin)
    
    # Check if we can solve this system (parallel lines don't intersect)
    if np.abs(np.linalg.det(A)) < 1e-10:
        ax.set_title("Parallel Lines: No Intersection")
    else:
        # Solve for parameters t1, t2
        t1, t2 = np.linalg.solve(A, b)
        
        # Calculate intersection point
        intersection = np.array(line1_origin) + t1 * np.array(line1_dir)
        
        # Plot intersection
        ax.scatter(intersection[0], intersection[1], color='red', s=100, 
                   label=f'Intersection at {intersection.round(2)}')
        ax.set_title(title)
    
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    plt.tight_layout()
    plt.savefig("../Outputs/Algebra/affine_intersection.png", dpi=300)
    return fig
    
# Interactive widget for intersection
@interact(
    l1_origin_x=(-3, 3), l1_origin_y=(-3, 3),
    l1_dir_x=(-2.0, 2.0, 0.5), l1_dir_y=(-2.0, 2.0, 0.5),
    l2_origin_x=(-3, 3), l2_origin_y=(-3, 3),
    l2_dir_x=(-2.0, 2.0, 0.5), l2_dir_y=(-2.0, 2.0, 0.5)
)
def update_intersection(
    l1_origin_x=0, l1_origin_y=0, l1_dir_x=1, l1_dir_y=1,
    l2_origin_x=0, l2_origin_y=2, l2_dir_x=1, l2_dir_y=0
):
    plot_intersection_lines(
        (l1_origin_x, l1_origin_y), (l1_dir_x, l1_dir_y),
        (l2_origin_x, l2_origin_y), (l2_dir_x, l2_dir_y)
    )



