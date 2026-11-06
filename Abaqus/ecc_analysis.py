
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("/Users/marigusdal/Documents/Advanced_product_sim/Abaqus/eccentricity_results")

# Initial physical lift height (mm)
h0 = 277.837

# Eccentricity Y : (Mises file, U3 file, CTF1 file)
cases = {
    0: ("max_mises_X0_Y0.csv", "U3_0x_0y.csv", "CTF1_0x_0y.csv"),
    100: ("max_mises_Y100.csv", "U3_100y.csv", "CTF1_100y.csv"),
    200: ("max_mises_Y200.csv", "U3_200y.csv", "CTF1_200y.csv"),
    300: ("max_mises_Y300.csv", "U3_300y.csv", "CTF1_300y.csv"),
    373: ("max_mises_Y373.csv", "U3_373y.csv", "CTF1_373y.csv")
}

# Longitudinal eccentricity cases
x_cases = {
    0: ("max_mises_X0_Y0.csv", "U3_0x_0y.csv", "CTF1_0x_0y.csv"),
    200: ("max_mises_X200_Y0.csv", "U3_200x_0y.csv", "CTF1_200x_0y.csv"),
    335: ("max_mises_X335_Y0.csv", "U3_335x_0y.csv", "CTF1_335x_0y.csv")
}

# Heights for sensitivity comparison
target_heights = [200, 600, 1000]


def interpolate(x, y, x_new):
    """Interpolate without extrapolating beyond available data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Average duplicate x values
    df = pd.DataFrame({"x": x, "y": y})
    df = df.groupby("x", as_index=False)["y"].mean()

    return np.interp(
        x_new, df["x"], df["y"],
        left=np.nan, right=np.nan
    )


data = {}

for e, (m_file, u_file, f_file) in cases.items():

    mises = pd.read_csv(folder / "mises" / m_file)
    u3 = pd.read_csv(folder / "u3" / u_file)
    force = pd.read_csv(folder / "ctf1" / f_file)

    u3 = u3[u3["Time"] >= 2.0].copy()
    force = force[force["Time"] >= 2.0].copy()

    # Convert U3 to lift height
    height = h0 + u3["U3"].to_numpy()

    # Associate stress and force with lift height
    stress_height = interpolate(
        u3["Time"], height, mises["Time"] + 2.0
    )

    force_height = interpolate(
        u3["Time"], height, force["Time"]
    )

    data[e] = {
        "stress_h": stress_height,
        "stress": mises["Max_Mises"].to_numpy(),
        "force_h": force_height,
        "force": np.abs(force["CTF1"].to_numpy())
    }


# --------------------------------------------------
# PROCESS X-DIRECTION DATA
# --------------------------------------------------

x_data = {}

for e, (m_file, u_file, f_file) in x_cases.items():

    mises = pd.read_csv(folder / "mises" / m_file)
    u3 = pd.read_csv(folder / "u3" / u_file)
    force = pd.read_csv(folder / "ctf1" / f_file)

    u3 = u3[u3["Time"] >= 2.0].copy()
    force = force[force["Time"] >= 2.0].copy()

    height = h0 + u3["U3"].to_numpy()

    stress_height = interpolate(
        u3["Time"], height, mises["Time"] + 2.0
    )

    force_height = interpolate(
        u3["Time"], height, force["Time"]
    )

    x_data[e] = {
        "stress_h": stress_height,
        "stress": mises["Max_Mises"].to_numpy(),
        "force_h": force_height,
        "force": np.abs(force["CTF1"].to_numpy())
    }



# --------------------------------------------------
# PLOT B: Relative von Mises stress sensitivity
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for h in target_heights:

    ref = interpolate(
        data[0]["stress_h"],
        data[0]["stress"],
        h
    )

    eccentricities = []
    changes = []

    for e, d in sorted(data.items()):

        stress = interpolate(
            d["stress_h"],
            d["stress"],
            h
        )

        if np.isfinite(ref) and ref > 0 and np.isfinite(stress):
            change = 100 * (stress / ref - 1)

            eccentricities.append(e)
            changes.append(change)

    if eccentricities:
        plt.plot(
            eccentricities, changes,
            marker="o", label=f"h = {h} mm"
        )

plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Transverse eccentricity Y [mm]")
plt.ylabel("Relative change in maximum Mises stress [%]")
plt.title("Stress sensitivity to eccentric loading")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("stress_sensitivity.png", dpi=300)
plt.show()


# --------------------------------------------------
# PLOT C: Actuator force versus lift height
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for e, d in sorted(data.items()):
    order = np.argsort(d["force_h"])

    plt.plot(
        d["force_h"][order],
        d["force"][order],
        label=f"Y = {e} mm"
    )

plt.xlabel("Lift height [mm]")
plt.ylabel("Actuator force [N]")
plt.title("Actuator force under eccentric loading")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actuator_eccentricity.png", dpi=300)
plt.show()


# --------------------------------------------------
# Print stress sensitivity at selected heights
# --------------------------------------------------

print("\nRelative stress change [%]")
print("Eccentricity", *[f"h={h}" for h in target_heights])

for e, d in sorted(data.items()):
    values = []

    for h in target_heights:
        ref = interpolate(
            data[0]["stress_h"],
            data[0]["stress"], h
        )

        stress = interpolate(
            d["stress_h"],
            d["stress"], h
        )

        if np.isfinite(ref) and ref > 0 and np.isfinite(stress):
            values.append(round(100 * (stress / ref - 1), 2))
        else:
            values.append(np.nan)

    print(e, *values)


# --------------------------------------------------
# PLOT D: Relative stress sensitivity in X
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for h in target_heights:

    ref = interpolate(
        x_data[0]["stress_h"],
        x_data[0]["stress"],
        h
    )

    eccentricities = []
    changes = []

    for e, d in sorted(x_data.items()):

        stress = interpolate(
            d["stress_h"],
            d["stress"],
            h
        )

        if np.isfinite(ref) and ref > 0 and np.isfinite(stress):
            change = 100 * (stress / ref - 1)

            eccentricities.append(e)
            changes.append(change)

    if eccentricities:
        plt.plot(
            eccentricities, changes,
            marker="o", label=f"h = {h} mm"
        )

plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Longitudinal eccentricity X [mm]")
plt.ylabel("Relative change in maximum Mises stress [%]")
plt.title("Stress sensitivity to longitudinal loading")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("stress_sensitivity_X.png", dpi=300)
plt.show()


# --------------------------------------------------
# PLOT E: Actuator force versus lift height (X)
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for e, d in sorted(x_data.items()):

    valid = (
        np.isfinite(d["force_h"]) &
        np.isfinite(d["force"])
    )

    if not np.any(valid):
        continue

    heights = d["force_h"][valid]
    forces = d["force"][valid]

    order = np.argsort(heights)

    plt.plot(
        heights[order],
        forces[order],
        label=f"X = {e} mm"
    )

plt.xlabel("Lift height [mm]")
plt.ylabel("Actuator force [N]")
plt.title("Actuator force under longitudinal loading")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actuator_eccentricity_X.png", dpi=300)
plt.show()


# --------------------------------------------------
# Print X-direction stress sensitivity
# --------------------------------------------------

print("\nLongitudinal stress sensitivity [%]")
print("Eccentricity X", *[f"h={h}" for h in target_heights])

for e, d in sorted(x_data.items()):

    values = []

    for h in target_heights:

        ref = interpolate(
            x_data[0]["stress_h"],
            x_data[0]["stress"],
            h
        )

        stress = interpolate(
            d["stress_h"],
            d["stress"],
            h
        )

        if np.isfinite(ref) and ref > 0 and np.isfinite(stress):
            values.append(round(100 * (stress / ref - 1), 2))
        else:
            values.append(np.nan)

    print(e, *values)
