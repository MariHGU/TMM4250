
import numpy as np
import matplotlib.pyplot as plt

import BeamModels_with_TODO as models
import SolverAlgorithms_with_TODO as algs
from pathlib import Path

# Folder for report figures
FIGURE_DIR = Path(__file__).resolve().parent / "report_figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Compare two mesh resolutions
mesh_sizes = [21, 41]

plt.figure(figsize=(10, 6))

for num_nodes in mesh_sizes:

    print(f"\nRunning analysis with {num_nodes} nodes")

    # Create a new model for each mesh
    model = models.DeepArchModel(num_nodes=num_nodes)

    # Run nonlinear arc-length analysis
    # Use more steps for the finer mesh
    max_steps = 600 if num_nodes == 21 else 800

    algs.solveArchLength(
        model,
        archLength=5.0,
        max_steps=max_steps,
        max_iter=30
    )

    # Extract results
    load_factors = np.array(model.load_history)
    crown_disp = np.array([
        u[model.plotDof - 1]
        for u in model.disp_history
    ])

    # Plot load-displacement curve
    plt.plot(
        -crown_disp,
        load_factors,
        label=f"{num_nodes} nodes ({num_nodes - 1} elements)"
    )

    # Print maximum load and corresponding displacement
    idx_max = np.argmax(load_factors)

    print(f"\n--- Results for {num_nodes} nodes ---")
    print(f"Maximum load factor: {load_factors[idx_max]:.6f}")
    print(f"Displacement at maximum load: {-crown_disp[idx_max]:.4f} mm")
    print(f"Final displacement: {-crown_disp[-1]:.4f} mm")
    print(f"Final load factor: {load_factors[-1]:.6f}")

plt.xlabel("Downward crown displacement [mm]")
plt.ylabel("Load factor λ")
plt.title("Deep arch – Mesh convergence comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig(
    FIGURE_DIR / "P2_04_MeshComparison_21vs41nodes.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

