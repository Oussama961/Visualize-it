from manim import *

class RiemannToIntegral(Scene):
    def construct(self):
        # Create a coordinate plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            x_length=7,
            y_range=[0, 16, 1],
            y_length=7
        ).to_edge(DOWN)
        
        # Plot the function f(x) = x^2 (Exemple)
        func = lambda x: x ** 2 # Create the function
        graph = plane.plot(func, x_range=[-4, 4], color=GREEN)
        
        # Add labels
        labels = plane.get_axis_labels(x_label="x", y_label=MathTex("f(x)"))
        graph_label = MathTex("f(x)=x^2").next_to(graph, RIGHT, buff=0.5).set_color(GREEN)
        
        # Display the plane, axes, and function
        self.play(DrawBorderThenFill(plane))
        self.play(Create(labels))
        self.play(Create(graph))
        self.play(Write(graph_label))
        self.wait(0.5)
        
        # Create initial Riemann rectangles
        rectangles = plane.get_riemann_rectangles(graph, x_range=[-4, 4], dx=1)
        self.play(FadeIn(rectangles))
        self.wait(0.5)
        
        # Smooth transition: dynamically update rectangles to a smaller dx, Calculate a new, smaller dx over time (decrease dx gradually)
        def update_rectangles(mob, dt):
            new_dx = max(0.1, mob.dx - 0.005)
            mob.become(plane.get_riemann_rectangles(graph, x_range=[-4, 4], dx=new_dx))
            mob.dx = new_dx

        rectangles.dx = 1  # Initialize a custom attribute for dx
        # Attach updater for smooth animation
        rectangles.add_updater(update_rectangles) 
        self.play(UpdateFromAlphaFunc(rectangles, lambda m, a: None), run_time=4)
        rectangles.clear_updaters()
        self.wait(1)

# The code sets up the coordinate plane and plots f(x)=x^2
# An initial set of Riemann rectangles is created using a larger width (dx=1) and then smoothly transformed into a finer approximation by gradually reducing the width via an updater.
# This animation effectively visualizes the limiting process where the Riemann sum converges to the definite integral.
