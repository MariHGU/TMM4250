
import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 60                  # Cylinder diameter [mm]
p = 200                 # Hydraulic pressure [bar]
m_test = 350            # Motorcycle mass [kg]
h0 = 277.84             # Initial lift height [mm]

# Maximum cylinder force [N], 1 bar = 0.1 N/mm²
F_max = p * 0.1 * np.pi * D**2 / 4

# Read Abaqus reports (time, value)
t_L, F_L = np.loadtxt("CTF1_loaded_abs.rpt", skiprows=2).T
t_S, F_S = np.loadtxt("CTF1_selfWeight_abs.rpt", skiprows=2).T
tu_L, u_L = np.loadtxt("U3_loaded.rpt", skiprows=2).T
tu_S, u_S = np.loadtxt("U3_selfWeight.rpt", skiprows=2).T

# Select lifting phase (t >= 2)
L = tu_L >= 2
S = tu_S >= 2

# Convert U3 to lift height and interpolate forces
h_L = h0 + u_L[L]
h_S = h0 + u_S[S]
F_L = np.interp(tu_L[L], t_L, F_L)
F_S = np.interp(tu_S[S], t_S, F_S)

# Compare forces at identical heights
h = np.linspace(max(200, h_L.min(), h_S.min()),
                min(1000, h_L.max(), h_S.max()), 200)

F_loaded = np.interp(h, h_L, F_L)
F_self = np.interp(h, h_S, F_S)

# Maximum payload [kg]
m_max = m_test * (F_max - F_self) / (F_loaded - F_self)

# Plot
plt.plot(h, m_max, label="D = 60 mm, p = 200 bar")
plt.xlabel("Lift height [mm]")
plt.ylabel("Maximum payload [kg]")
plt.title("Lift capacity as a function of lift height")
plt.grid(True)
plt.legend()
plt.show()

print(f"Maximum cylinder force: {F_max/1000:.2f} kN")
print(f"Minimum lift capacity: {m_max.min():.1f} kg")
