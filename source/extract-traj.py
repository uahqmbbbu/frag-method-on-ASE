import sys
import numpy as np
import MDAnalysis as mda
from ase.io.trajectory import Trajectory

if len(sys.argv) != 3:
    print("Usage: python extract-traj.py <input.traj> <output.dcd>")
    sys.exit(1)

traj_file = sys.argv[1]
opt_traj_file = sys.argv[2]

traj = Trajectory(traj_file)
n_atoms = len(traj[0])
frames = np.array([atoms.get_positions() for atoms in traj], dtype=np.float32)

u = mda.Universe.empty(n_atoms, trajectory=True)
u.load_new(frames)

with mda.Writer(opt_traj_file, n_atoms=n_atoms) as w:
    for ts in u.trajectory:
        w.write(u.atoms)

print(f'Done: {len(frames)} frames written to {opt_traj_file}')
