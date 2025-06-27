import sympy as sp

def double_integral(f, limits):
    """Compute a double integral using SymPy."""
    x, y = sp.symbols('x y')
    inner = sp.integrate(f, limits[1])
    outer = sp.integrate(inner, limits[0])
    return outer