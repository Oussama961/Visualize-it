from ipywidgets import interact, FloatSlider
import numpy as np
import matplotlib.pyplot as plt

# Import from our modules
from barycenter_core import compute_barycenter, compute_leibniz_function
from barycenter_plots import plot_barycenter, configure_seaborn

def interactive_barycenter():
    """Widget to adjust weights and see barycenter changes."""
    # Default points
    points = [(1, 2), (-1, 3), (2, -1)]
    
    # Sliders for weights
    weight_sliders = [FloatSlider(min=-2, max=2, value=1, description=f'λ{i+1}') 
                      for i in range(len(points))]
    
    @interact(**{f'w{i}': slider for i, slider in enumerate(weight_sliders)})
    def update_plot(**weights):
        weights = list(weights.values())
        try:
            plot_barycenter(points, weights)
        except ValueError:
            print("Total weight is zero! Barycenter undefined.")

def interactive_leibniz():
    """Show Leibniz vector vanishing at barycenter."""
    points = [(1, 1), (3, 2)]
    weights = [1, 1]
    G = compute_barycenter(points, weights)
    
    @interact(M_x=(-5.0, 5.0), M_y=(-5.0, 5.0))
    def update_leibniz(M_x=0.0, M_y=0.0):
        plot_barycenter(points, weights, show_leibniz=True, reference_M=(M_x, M_y))
        if np.allclose((M_x, M_y), G):
            print("Leibniz vector is zero at barycenter G!")
        else:
            M_to_G_dist = np.sqrt((M_x - G[0])**2 + (M_y - G[1])**2)
            print(f"Distance from M to G: {M_to_G_dist:.2f}")

# Additional example showing barycentric coordinates
def interactive_barycentric_coords():
    """Demonstrate barycentric coordinates in a triangle."""
    # Triangle vertices
    triangle = [(0, 0), (4, 0), (2, 3)]
    
    @interact(λ1=FloatSlider(min=0, max=1, step=0.01, value=1/3),
              λ2=FloatSlider(min=0, max=1, step=0.01, value=1/3))
    def update_coords(λ1=1/3, λ2=1/3):
        λ3 = 1 - λ1 - λ2
        if λ3 < 0:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "Invalid barycentric coordinates\nλ1 + λ2 > 1", 
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.show()
            return
            
        configure_seaborn()
        weights = [λ1, λ2, λ3]
        fig, ax = plt.subplots()
        
        # Draw the triangle
        triangle_x = [p[0] for p in triangle] + [triangle[0][0]]
        triangle_y = [p[1] for p in triangle] + [triangle[0][1]]
        ax.plot(triangle_x, triangle_y, 'k-', alpha=0.5)
        
        # Plot vertices with weights
        for i, ((x, y), w) in enumerate(zip(triangle, weights)):
            ax.scatter(x, y, s=300, label=f'A{i+1} (λ={w:.2f})')
        
        # Compute and plot barycenter
        G = compute_barycenter(triangle, weights)
        ax.scatter(*G, s=200, color='red', marker='X', label=f'Point at {G}')
        
        ax.set_title(f"Barycentric Coordinates: (λ1={λ1:.2f}, λ2={λ2:.2f}, λ3={λ3:.2f})")
        ax.legend()
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 4)
        plt.tight_layout()
        plt.show() 