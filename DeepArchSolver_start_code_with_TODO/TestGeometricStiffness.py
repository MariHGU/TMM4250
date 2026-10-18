
import numpy as np
from CorotBeam_with_TODO import beam2corot_Ke_and_Fe

# A single beam element
ex = np.array([0.0, 100.0])
ey = np.array([0.0, 0.0])
ep = np.array([210000.0, 10.0, 83.3333])

# Nonzero translations and rotations
u = np.array([0.5, -2.0, 0.02, 1.0, 3.0, -0.03])

# Analytical tangent stiffness
K, f = beam2corot_Ke_and_Fe(ex, ey, ep, u)

# Numerical derivative of internal forces
eps = 1e-5
K_numerical = np.zeros((6, 6))

for i in range(6):
    du = np.zeros(6)
    du[i] = eps

    _, f_plus = beam2corot_Ke_and_Fe(
        ex, ey, ep, u + du
    )
    _, f_minus = beam2corot_Ke_and_Fe(
        ex, ey, ep, u - du
    )

    K_numerical[:, i] = (
        f_plus - f_minus
    ) / (2 * eps)

error = np.linalg.norm(K - K_numerical)
relative_error = error / np.linalg.norm(K_numerical)

print(f"Absolute error: {error:.6e}")
print(f"Relative error: {relative_error:.6e}")

assert relative_error < 1e-6
print("Tangent stiffness test PASSED")
