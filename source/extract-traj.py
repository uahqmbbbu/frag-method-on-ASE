import sys
from ase.io.trajectory import Trajectory
from ase.io import write

if len(sys.argv) != 3:
    print("Usage: python extract-traj.py <input.traj> <output.xyz>")
    sys.exit(1)

traj_file = sys.argv[1]
opt_traj_file = sys.argv[2]

traj = Trajectory(traj_file)
for i, atoms in enumerate(traj):
    write(opt_traj_file, atoms, append=True)
print(f'Done: {len(traj)} frames written to {opt_traj_file}')
