from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact

def interactive_scalar_multiplication():
    """Interactive demo of scalar multiplication."""
    v = widgets.FloatSlider(value=1, min=-5, max=5, step=0.5, description='Scalar (α):')
    u = widgets.FloatSlider(value=1, min=-5, max=5, step=0.5, description='Vector x:')
    
    @widgets.interact(alpha=v, x=u)
    def update_plot(alpha, x):
        plot_scalar_multiplication(alpha, (x, x))
    

def interactive_linear_combination():
    """Interactive demo for linear combinations."""
    coeff1 = widgets.FloatSlider(min=-2, max=2, step=0.5, value=1, description='Coeff 1:')
    coeff2 = widgets.FloatSlider(min=-2, max=2, step=0.5, value=1, description='Coeff 2:')
    
    @widgets.interact(c1=coeff1, c2=coeff2)
    def update_linear_combination(c1, c2):
        vectors = [(1, 2), (3, 1)]
        plot_linear_combination(vectors, [c1, c2])
    
def interactive_product_space():
    """Interactive demo for E1 × E2 operations."""
    x1 = widgets.FloatSlider(min=-3, max=3, value=1, description='x1 (E1):')
    x2 = widgets.FloatSlider(min=-3, max=3, value=2, description='x2 (E2):')
    alpha = widgets.FloatSlider(min=-2, max=2, value=1, description='α:')
    
    @widgets.interact(x1=x1, x2=x2, alpha=alpha)
    def update_product_space(x1, x2, alpha):
        fig1 = plot_vector_addition_product(v1=(x1, x2), v2=(0.5, -1))
        fig2 = plot_scalar_multiplication_product(alpha=alpha, v=(x1, x2))
        plt.show()


def interactive_subspace_check():
    """Interactive demo to test subspace properties."""
    x_slider = widgets.FloatSlider(min=-3, max=3, value=1, description='x:')
    y_slider = widgets.FloatSlider(min=-3, max=3, value=2, description='y:')
    
    @widgets.interact(x=x_slider, y=y_slider)
    def update_subspace_check(x, y):
        # Check if (x, y) lies on a subspace (e.g., y = 2x)
        is_subspace = abs(y - 2*x) < 1e-9  # Tolerance for floating point
        
        if is_subspace:
            fig = plot_subspace_example(vectors=[(x, y)], 
                    title=f"Valid Subspace: y = 2x (Includes ({x}, {y}))")
        else:
            fig = plot_non_subspace_example(vectors=[(x, y)], 
                    title=f"Not a Subspace: ({x}, {y}) not on y = 2x")
        plt.show()


def interactive_span_explorer():
    """Interactive demo for Vect(A)."""
    x1 = widgets.FloatSlider(min=-2, max=2, value=1, description='v1_x:')
    y1 = widgets.FloatSlider(min=-2, max=2, value=1, description='v1_y:')
    x2 = widgets.FloatSlider(min=-2, max=2, value=2, description='v2_x:')
    y2 = widgets.FloatSlider(min=-2, max=2, value=-1, description='v2_y:')
    
    @widgets.interact(v1_x=x1, v1_y=y1, v2_x=x2, v2_y=y2)
    def update_span(v1_x, v1_y, v2_x, v2_y):
        plot_span_A(vectors=[(v1_x, v1_y), (v2_x, v2_y)])


def interactive_finite_span():
    """Interactive demo for finite spans."""
    x1 = widgets.FloatSlider(min=-2, max=2, value=1, description='a₁_x:')
    y1 = widgets.FloatSlider(min=-2, max=2, value=2, description='a₁_y:')
    x2 = widgets.FloatSlider(min=-2, max=2, value=-1, description='a₂_x:')
    y2 = widgets.FloatSlider(min=-2, max=2, value=1, description='a₂_y:')
    @widgets.interact(a1_x=x1, a1_y=y1, a2_x=x2, a2_y=y2)
    def update_finite_span(a1_x, a1_y, a2_x, a2_y):
        plot_finite_span(vectors=[(a1_x, a1_y), (a2_x, a2_y)])


def interactive_subspace_sum_2d():
    """Interactive demo for 2D subspace sums."""
    slope1 = widgets.FloatSlider(min=-2, max=2, value=1, description='Slope F₁:')
    slope2 = widgets.FloatSlider(min=-2, max=2, value=-1, description='Slope F₂:')    
    @widgets.interact(slope1=slope1, slope2=slope2)
    def update_subspace_sum(slope1, slope2):
        fig = plot_subspace_sum_2d(subspaces=[(slope1, 0), (slope2, 0)], title=f"Sum of F₁ (y={slope1}x) + F₂ (y={slope2}x)")
        plt.show()


def interactive_direct_sum_2d():
    """Check if two lines are supplementary subspaces."""
    slope1 = widgets.FloatSlider(min=-2, max=2, value=0, description='Slope F:')
    slope2 = widgets.FloatSlider(min=-2, max=2, value=float('inf'), description='Slope G:')
    
    @widgets.interact(slope1=slope1, slope2=slope2)
    def update_direct_sum(slope1, slope2):
        # Check if F ∩ G = {0} and F + G = ℝ²
        is_supplementary = (slope1 != slope2) or (slope1 == 0 and slope2 == float('inf'))        
        if is_supplementary:
            fig = plot_direct_sum_2d(subspace1=lambda x: slope1*x, subspace2=lambda x: slope2*x, title=f"E = F ⊕ G: {is_supplementary}")
        else:
            fig = plot_subspace_sum_2d(  # From previous code
                subspaces=[(slope1, 0), (slope2, 0)],
                title=f"Not Supplementary: F ∩ G ≠ {{0}}"
            )
        plt.show()

def plot_translation(point_A, vector_u):
    """Visualize translating a point by a vector in affine space."""
    configure_seaborn()
    fig, ax = plt.subplots()
    
    # Original point A
    ax.scatter(*point_A, s=100, color='blue', label=f'Point A = {point_A}')
    
    # Translated point B = A + u
    point_B = (point_A[0] + vector_u[0], point_A[1] + vector_u[1])
    ax.scatter(*point_B, s=100, color='green', label=f'B = A + u = {point_B}')
    
    # Vector u from A to B
    ax.quiver(*point_A, vector_u[0], vector_u[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.005, label=f'Vector u = {vector_u}')
    
    ax.set_title(f"Translation by u = {vector_u}")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.show()

# Interactive widget
@interact(
    A_x=(0, 10), A_y=(0, 10),
    u_x=(-5, 5), u_y=(-5, 5)
)
def update_translation(A_x=1, A_y=1, u_x=2, u_y=3):
    plot_translation((A_x, A_y), (u_x, u_y))