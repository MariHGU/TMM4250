
import numpy as np
import matplotlib.pyplot as plt

# Abaqus results
height = np.array([205.51, 405.73, 606.50, 799.67, 1017.00])
eigenvalues = np.array([13.556, 18.597, 20.842, 23.124, 26.415])

# Reference motorcycle weight [N]
F_ref = 350 * 9.81

# Critical buckling loads [kN]
F_cr = eigenvalues * F_ref / 1000

# Interpolation function (h in mm, output in kN)
def buckling_load(h):
    return np.interp(h, height, F_cr)

# Plot from 205.51 to 1017 mm
h_plot = np.linspace(height[0], height[-1], 300)

plt.plot(h_plot, buckling_load(h_plot), label="Interpolated buckling load")
plt.scatter(height, F_cr, label="Abaqus results")
plt.xlabel("Lift height [mm]")
plt.ylabel("Critical buckling load [kN]")
plt.grid(True)
plt.legend()
plt.show()

# Test: critical buckling load at 500 mm
print(f"Buckling load at 500 mm: {buckling_load(500):.2f} kN")
