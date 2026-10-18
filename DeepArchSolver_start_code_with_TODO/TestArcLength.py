
import numpy as np
import matplotlib.pyplot as plt

import BeamModels_with_TODO as models
import SolverAlgorithms_with_TODO as algs
from pathlib import Path

# Folder for report figures
FIGURE_DIR = Path(__file__).resolve().parent / "report_figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Create deep arch model
model = models.DeepArchModel(num_nodes=41)

# Run arc-length analysis
algs.solveArchLength(
    model,
    archLength=5.0,
    max_steps=800,
    max_iter=30
)

# Extract load factors and crown displacements
load_factors = np.array(model.load_history)
crown_disp = np.array([
    u[model.plotDof - 1]
    for u in model.disp_history
])


# Plot load-displacement response
plt.figure(figsize=(9, 6))
plt.plot(-crown_disp, load_factors, "b.-", markersize=2)

plt.xlabel("Downward crown displacement [mm]")
plt.ylabel("Load factor λ")
plt.title("Deep arch - Nonlinear arc-length response")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "P2_02_NonlinearResponse_41nodes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# Print final results
print("\n--- Arc-length results ---")
print(f"Converged steps: {len(load_factors) - 1}")
print(f"Final load factor: {load_factors[-1]:.6f}")
print(f"Final crown displacement: {crown_disp[-1]:.4f} mm")


# Plot deformed shapes at selected equilibrium states

# Select states from the solution history
idx_initial = 0
idx_max_load = np.argmax(load_factors)
idx_max_disp = np.argmin(crown_disp)
idx_final = len(load_factors) - 1

selected_steps = [
    (idx_initial, "Initial state"),
    (idx_max_load, "Maximum positive load"),
    (idx_max_disp, "Maximum downward displacement"),
    (idx_final, "Final state")
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (step, title) in zip(axes.flat, selected_steps):

    # Get displacement vector at selected step
    u = model.disp_history[step]

    # Calculate deformed coordinates
    deformed = model.coords.copy()
    deformed[:, 0] += u[0::3]
    deformed[:, 1] += u[1::3]

    # Plot original and deformed geometry
    ax.plot(
        model.coords[:, 0],
        model.coords[:, 1],
        "k--",
        label="Undeformed"
    )

    ax.plot(
        deformed[:, 0],
        deformed[:, 1],
        "b-",
        label="Deformed"
    )

    ax.set_title(
        f"{title}\n"
        f"Step {step}, "
        f"Lambda = {load_factors[step]:.2f}"
    )

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()


plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "P2_03_DeformedShapes_41nodes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

