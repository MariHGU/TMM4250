
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import BeamModels_with_TODO as models
import SolverAlgorithms_with_TODO as algs

# Folder for report figures
FIGURE_DIR = Path(__file__).resolve().parent / "report_figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Create deep arch model
model = models.DeepArchModel(num_nodes=21)

# Run linear analysis for lambda = 1
algs.solveLinearSteps(
    model,
    load_steps=1.0,
    max_steps=2
)

# Extract results
u = model.disp_history[-1]
crown_disp = u[model.plotDof - 1]

print("\n--- Linear analysis results ---")
print(f"Reference load: {model.F0:.4f} N")
print(f"Crown displacement: {crown_disp:.4f} mm")
print(f"Load factor: {model.load_history[-1]:.2f}")

# Calculate deformed coordinates
deformed = model.coords.copy()
deformed[:, 0] += u[0::3]
deformed[:, 1] += u[1::3]

# Plot undeformed and deformed arch
plt.figure(figsize=(10, 6))

plt.plot(
    model.coords[:, 0], model.coords[:, 1],
    "k--", label="Undeformed"
)
plt.plot(
    deformed[:, 0], deformed[:, 1],
    "b-", label="Linear deformation"
)

plt.axis("equal")
plt.xlabel("X [mm]")
plt.ylabel("Y [mm]")
plt.title("Deep arch - Linear analysis (21 nodes)")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "P2_01_LinearAnalysis_21nodes.png",
    dpi=300, bbox_inches="tight"
)
plt.show()
