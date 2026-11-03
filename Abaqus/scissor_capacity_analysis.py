"""Scissor-lift actuator-limited payload capacity from Abaqus XY .rpt data.

Requires: numpy, matplotlib
Place this script alongside the four .rpt files, then run:
    python scissor_capacity_analysis.py

Units: input forces N, displacement and heights mm, pressure bar.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FILES = {
    'f_loaded': 'CTF1_loaded_abs.rpt',
    'f_empty': 'CTF1_selfWeight_abs.rpt',
    'u_loaded': 'U3_loaded(1).rpt',
    'u_empty': 'U3_selfWeight(1).rpt',
}
BORE_MM = 60.0
PRESSURE_BAR = 200.0
PRESSURES_BAR = [150.0, 200.0, 250.0]
TEST_MASS_KG = 350.0
# Calibrated from measured lift heights: 185.17 mm at U3=-92.6695 mm,
# and 1002.96 mm at U3=725.118 mm for the loaded run.
# This is the initially measured lift HEIGHT, not the initial node z-coordinate.
INITIAL_LIFT_HEIGHT_MM = 277.84
LIFT_STEP_START_TIME = 2.0  # analysis lowering finishes and raising starts here
REQUIRED_MIN_MM, REQUIRED_MAX_MM = 200.0, 1000.0


def parse_report(path):
    """Read Abaqus XY report rows with two numeric columns; skip header/footer."""
    num = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?'
    expr = re.compile(r'^\s*(' + num + r')\s+(' + num + r')\s*$')
    data = []
    for line in path.read_text(errors='replace').splitlines():
        m = expr.match(line)
        if m:
            data.append((float(m.group(1).replace('D','E').replace('d','e')),
                         float(m.group(2).replace('D','E').replace('d','e'))))
    if not data:
        raise ValueError(f'No X-Y numbers found in {path}')
    arr = np.asarray(data, dtype=float)
    # At step boundaries Abaqus can store duplicate time stamps.
    # Keep the last occurrence and sort ascending.
    unique_times, idx_rev = np.unique(arr[::-1, 0], return_index=True)
    idx = len(arr) - 1 - idx_rev
    order = np.argsort(arr[idx, 0])
    return arr[idx, 0][order], arr[idx, 1][order]


def lift_branch(force_path, u_path):
    tf, f = parse_report(force_path)
    tu, u = parse_report(u_path)
    # Only lifting (step t >= 2), not previous lowering/gravity history.
    mask = (tu >= LIFT_STEP_START_TIME - 1e-9)
    tu, u = tu[mask], u[mask]
    if len(tu) < 2:
        raise ValueError(f'Not enough lifting frames in {u_path}')
    valid = (tu >= tf.min()) & (tu <= tf.max())
    tu, u = tu[valid], u[valid]
    force = np.interp(tu, tf, f)
    height = INITIAL_LIFT_HEIGHT_MM + u
    # Sort by height: comparing force at the SAME HEIGHT, not same time.
    order = np.argsort(height)
    height, force = height[order], force[order]
    if np.any(np.diff(height) <= 0):
        raise ValueError('Lift height is not strictly increasing on selected branch.')
    return height, force


def cylinder_force_newtons(pressure_bar):
    area_mm2 = np.pi * BORE_MM**2 / 4
    # 1 bar = 0.1 N/mm². Ideal piston extension, no losses.
    return pressure_bar * 0.1 * area_mm2


def main():
    paths = {key: ROOT / name for key, name in FILES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')

    h_load, f_load = lift_branch(paths['f_loaded'], paths['u_loaded'])
    h_empty, f_empty = lift_branch(paths['f_empty'], paths['u_empty'])
    h_min = max(REQUIRED_MIN_MM, h_load.min(), h_empty.min())
    h_max = min(REQUIRED_MAX_MM, h_load.max(), h_empty.max())
    if h_max <= h_min:
        raise ValueError('Loaded and unloaded lifting ranges do not overlap.')
    if h_min > REQUIRED_MIN_MM:
        print(f'WARNING: No common data below {h_min:.2f} mm. '
              'Do NOT report capacity at 200 mm without an additional run.')

    # Sample at 5 mm spacing plus common-range boundaries.
    heights = np.unique(np.r_[h_min, np.arange(np.ceil(h_min / 5) * 5,
                                                h_max + 1e-9, 5.0), h_max])
    loaded = np.interp(heights, h_load, f_load)
    empty = np.interp(heights, h_empty, f_empty)
    difference = loaded - empty
    if np.any(difference <= 0):
        raise ValueError('Unexpected load contribution <= 0. Check sign, branches, units.')

    def capacity_kg(pressure_bar):
        return TEST_MASS_KG * (cylinder_force_newtons(pressure_bar) - empty) / difference

    p_col = f'capacity_{PRESSURE_BAR:g}bar_kg'
    columns = [heights, loaded/1000, empty/1000, difference/1000]
    headers = ['height_mm', 'force_loaded_kN', 'force_selfweight_kN', 'force_test_payload_kN']
    for p in sorted(set(PRESSURES_BAR + [PRESSURE_BAR])):
        columns.append(capacity_kg(p))
        headers.append(f'capacity_{p:g}bar_kg')
    out_csv = ROOT / 'scissor_capacity_corrected.csv'
    np.savetxt(out_csv, np.column_stack(columns), delimiter=',',
               header=','.join(headers), comments='', fmt='%.5f')

    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.plot(heights, loaded/1000, label='350 kg motorcycle + self-weight')
    ax.plot(heights, empty/1000, label='Self-weight only')
    ax.set(xlabel='Lift height (mm)', ylabel='Required cylinder force (kN)',
           title='Actuator force during lifting')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/'scissor_force_vs_height_corrected.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5.5))
    for p in PRESSURES_BAR:
        ax.plot(heights, capacity_kg(p), label=f'{p:g} bar (D = {BORE_MM:g} mm)')
    ax.axhline(TEST_MASS_KG, linestyle='--', color='gray', label='350 kg reference')
    ax.set(xlabel='Lift height (mm)', ylabel='Actuator-limited payload (kg)',
           title='Maximum payload versus lift height (ideal extension)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/'scissor_capacity_vs_height_corrected.png', dpi=180)
    plt.close(fig)

    cap = capacity_kg(PRESSURE_BAR)
    print(f'Cylinder: D = {BORE_MM:.0f} mm, p = {PRESSURE_BAR:.0f} bar, '
          f'Fmax = {cylinder_force_newtons(PRESSURE_BAR)/1000:.3f} kN')
    print(f'Comparable lift-height interval: {h_min:.2f} to {h_max:.2f} mm')
    print(f'Minimum predicted capacity in interval: {cap.min():.1f} kg')
    for h in [220,250,300,500,750,1000]:
        if h_min <= h <= h_max:
            print(f'h={h:4d} mm: loaded={np.interp(h,heights,loaded)/1000:.2f} kN, '
                  f'empty={np.interp(h,heights,empty)/1000:.2f} kN, '
                  f'capacity={np.interp(h,heights,cap):.1f} kg')
    print('Saved:', out_csv.name, 'and two PNG plots')


if __name__ == '__main__':
    main()
