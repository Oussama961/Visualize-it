from bases.generating_families import plot_generating_family_2d
from bases.linear_independence import plot_linear_independence
from bases.bases_and_dimension import plot_basis_3d
from bases.incomplete_basis import plot_incomplete_basis

# Render all visuals in a grid
fig = plt.figure(figsize=(15, 10))

# Generating family example
plt.subplot(2, 2, 1)
plot_generating_family_2d((1, 2), (3, 1))

# Linear independence example
plt.subplot(2, 2, 2)
plot_linear_independence((1, 2), (2, 4))

# Basis and dimension example
plt.subplot(2, 2, 3, projection='3d')
plot_basis_3d((1,0,0), (0,1,0), (0,0,1))

# Incomplete basis theorem
plt.subplot(2, 2, 4)
plot_incomplete_basis()

plt.tight_layout()