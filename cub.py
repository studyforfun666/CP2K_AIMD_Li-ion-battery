import numpy as np
import MDAnalysis as mda

# -------------------- User Inputs -------------------- #
xyz_file = "test.xyz"          # Replace with your actual file
atom_selection = "name Li"           # e.g. "all", "name O", "name H"
box = [15.378389, 10.741209, 14.778325]  # [a, b, c] in Å
nbins = [30, 21, 29]                # Grid points along x, y, z
origin = [0.0, 0.0, 0.0]            # Cube origin
output_cube = "density_map.cube"
# ----------------------------------------------------- #

# Calculate grid spacing (in Å)
spacing = [box[i] / nbins[i] for i in range(3)]

# Load trajectory
u = mda.Universe(xyz_file, format="XYZ")
u.dimensions = box + [90.0, 90.0, 90.0]  # Box angles

# Select atoms
selection = u.select_atoms(atom_selection)

# Initialize 3D grid
density = np.zeros(nbins)

# Accumulate density
for ts in u.trajectory:
    positions = selection.positions

    indices = np.floor([
        (positions[:, 0] - origin[0]) / spacing[0],
        (positions[:, 1] - origin[1]) / spacing[1],
        (positions[:, 2] - origin[2]) / spacing[2]
    ]).astype(int).T

    for ix, iy, iz in indices:
        if 0 <= ix < nbins[0] and 0 <= iy < nbins[1] and 0 <= iz < nbins[2]:
            density[ix, iy, iz] += 1

# Normalize
density /= len(u.trajectory)

# ------------ Save Cube File for VESTA --------------- #
def save_cube(filename, data, origin, spacing):
    nx, ny, nz = data.shape

    # Convert Å to Bohr (VESTA expects Bohr grid spacing)
    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
    spacing_bohr = [s * ANGSTROM_TO_BOHR for s in spacing]
    origin_bohr = [o * ANGSTROM_TO_BOHR for o in origin]

    with open(filename, "w") as f:
        f.write("Spatial probability density cube\n")
        f.write("VESTA-compatible cube (grid in Bohr)\n")
        f.write(f"{1:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}\n")
        f.write(f"{nx:5d} {spacing_bohr[0]:12.6f} 0.000000 0.000000\n")
        f.write(f"{ny:5d} 0.000000 {spacing_bohr[1]:12.6f} 0.000000\n")
        f.write(f"{nz:5d} 0.000000 0.000000 {spacing_bohr[2]:12.6f}\n")

        # Dummy atom (atomic number 0)
        f.write(f"{0:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")

        # Flatten data: X fastest, Z slowest
        flat = data.transpose(0, 1, 2).flatten()
        for i in range(0, len(flat), 6):
            f.write(" ".join(f"{val:13.5e}" for val in flat[i:i+6]) + "\n")

# Save it
save_cube(output_cube, density, origin, spacing)
print(f"✅ Cube file saved as '{output_cube}' with exact box size.")

