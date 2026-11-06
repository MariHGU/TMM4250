
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Folder containing your exported stress files
folder = Path("/Users/marigusdal/Documents/Advanced_product_sim/Abaqus/mises_res")

# Read all eccentricity cases
files = sorted(folder.glob("max_mises_*.csv"))

data = {}

for file in files:
    name = file.stem.replace("max_mises_", "")
    df = pd.read_csv(file)

    data[name] = df.sort_values("Time")

# Centered reference
reference = data["X0_Y0"]

# Plot maximum von Mises stress
plt.figure(figsize=(9, 5))

for name, df in data.items():
    plt.plot(
        df["Time"],
        df["Max_Mises"],
        label=name
    )

plt.xlabel("Step time")
plt.ylabel("Maximum von Mises stress [MPa]")
plt.title("Effect of eccentric loading on maximum stress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare stress with centered reference
print("\nStress sensitivity relative to centered loading:\n")

for name, df in data.items():

    # Compare values at matching times
    combined = pd.merge(
        reference[["Time", "Max_Mises"]],
        df[["Time", "Max_Mises"]],
        on="Time",
        suffixes=("_ref", "_ecc")
    )

    if combined.empty:
        print(name, ": No matching time values")
        continue

    change = 100 * (
        combined["Max_Mises_ecc"]
        / combined["Max_Mises_ref"] - 1
    )

    # Exclude zero-stress reference frames
    change = change.replace(
        [float("inf"), -float("inf")], float("nan")
    ).dropna()

    print(
        "{}: Max stress = {:.2f} MPa, "
        "Max relative increase = {:.2f}%".format(
            name,
            df["Max_Mises"].max(),
            change.max()
        )
    )
