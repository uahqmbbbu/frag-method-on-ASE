#!/usr/bin/env python3
"""QMMM MD entry point."""

import argparse

from ase import Atoms
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kcal, mol

from calculators import QMMM, MMCalculator, QMCalculator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QMMM molecular dynamics.")

    # === System ===
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument("-f", "--file", type=str, required=True,
                           help="Input structure file (PDB, XYZ, ...).")
    sys_group.add_argument("--format", type=str, default=None,
                           help="File format (overrides auto-detection).")

    # === QM region ===
    qm_group = parser.add_argument_group("QM region")
    qm_group.add_argument("--selection", type=int, nargs="+", required=True,
                          help="Atom indices (0-based) in the QM region.")
    qm_group.add_argument("--qm-pdb", type=str, required=True,
                          help="PDB file for QM calculator topology.")
    qm_group.add_argument("--qm-model", type=str, required=True,
                          help="TorchScript MACE model file (.pt).")
    qm_group.add_argument("--qm-precision", type=str, default="fp32",
                          choices=["fp32", "fp64"],
                          help="QM calculation precision (default: fp32).")
    qm_group.add_argument("--qm-device", type=str, default="cpu",
                          choices=["cpu", "cuda"],
                          help="QM device (default: cpu).")
    qm_group.add_argument("--qm-batch-size", type=int, default=64,
                          help="QM fragment batch size (default: 64).")
    qm_group.add_argument("--vacuum", action="store_true", default=False,
                          help="Remove net translation of QM forces.")

    # === MD ===
    md_group = parser.add_argument_group("MD")
    md_group.add_argument("--timestep", type=float, default=1.0,
                          help="Integration timestep in fs (default: 1.0).")
    md_group.add_argument("--temperature", type=float, default=300,
                          help="Target temperature in K (default: 300).")
    md_group.add_argument("--friction", type=float, default=0.01,
                          help="Langevin friction in 1/fs (default: 0.01).")
    md_group.add_argument("--nsteps", type=int, default=1000,
                          help="Number of MD steps (default: 1000).")

    # === Output ===
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--traj", type=str, default=None,
                           help="Trajectory output file.")
    out_group.add_argument("--traj-interval", type=int, default=10,
                           help="Trajectory write interval (default: 10).")
    out_group.add_argument("--log-interval", type=int, default=1,
                           help="Logging interval (default: 1).")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # === System ===
    atoms: Atoms = read(args.file, format=args.format)

    # === Calculators ===
    qmcalc = QMCalculator(pdb_file=args.qm_pdb,
                          model_path=args.qm_model,
                          precision=args.qm_precision,
                          device=args.qm_device,
                          batch_size=args.qm_batch_size)
    mmcalc = MMCalculator()

    qmmm = QMMM(
        selection=args.selection,
        qmcalc=qmcalc,
        mmcalc1=mmcalc,
        mmcalc2=mmcalc,
        vacuum=args.vacuum,
    )
    atoms.calc = qmmm

    # === MD ===
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature * kcal / mol)

    dyn = Langevin(
        atoms,
        timestep=args.timestep * fs,
        temperature_K=args.temperature,
        friction=args.friction / fs,
    )

    # === Attach observers ===
    if args.traj:
        from ase.io.trajectory import TrajectoryWriter
        traj = TrajectoryWriter(args.traj, mode="w")
        dyn.attach(traj.write, interval=args.traj_interval, atoms=atoms)

    def log():
        epot = atoms.get_potential_energy()
        print(f"Step {dyn.nsteps:6d}  Epot = {epot:12.4f} eV")

    dyn.attach(log, interval=args.log_interval)

    # === Run ===
    print(f"Running {args.nsteps} steps of Langevin dynamics...")
    dyn.run(args.nsteps)
    print("Done.")


if __name__ == "__main__":
    main()
