#!/usr/bin/env python3
"""QMMM MD entry point.  Run from any directory::

    python source/main.py -f full.pdb --qm-pdb qm.pdb --qm-model model.pt ...
"""

import argparse
import os
import sys
import time

# ensure source/ and source/lib/ are importable from any working directory
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_lib_dir = os.path.join(_this_dir, "..", "lib")
if _lib_dir not in sys.path:
    sys.path.insert(0, _lib_dir)

from ase import Atoms
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kcal, mol

from calculators import QMMM, MMCalculator, QMCalculator
from calculators.prepare_mm import prepare


def auto_selection(full_pdb: str, qm_pdb: str) -> list[int]:
    """Match QM-PDB atoms to full-system PDB atoms by number+type+residue.

    Returns 0-based indices into the full system.
    """
    full = read(full_pdb)
    qm = read(qm_pdb)

    indices = []
    qm_i = 0
    for fi in range(len(full)):
        if qm_i >= len(qm):
            break
        if (full.numbers[fi] == qm.numbers[qm_i]
            and full.arrays["atomtypes"][fi] == qm.arrays["atomtypes"][qm_i]
            and full.arrays["residuenumbers"][fi] == qm.arrays["residuenumbers"][qm_i]):
            indices.append(fi)
            qm_i += 1

    if len(indices) != len(qm):
        raise RuntimeError(
            f"auto-selection matched {len(indices)} of {len(qm)} QM atoms. "
            "Ensure --qm-pdb atoms exist in -f PDB in the same order."
        )
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QMMM molecular dynamics.")

    # === System ===
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument("-f", "--file", type=str, required=True,
                           help="Input structure file (PDB, XYZ, ...).")
    sys_group.add_argument("--format", type=str, default=None,
                           help="File format (overrides auto-detection).")

    qm_group = parser.add_argument_group("QMCalc")
    qm_group.add_argument("--selection", type=int, nargs="+", default=None,
                          help="QM atom indices (0-based). If omitted, auto-detected from --qm-pdb.")
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

    mm_group = parser.add_argument_group("MM")
    mm_group.add_argument("--force-fields", type=str, nargs="+", required=True,
                          help="Force field XML files (e.g. amber14-all.xml tip3p.xml).")
    mm_group.add_argument("--mm-dir", type=str, default="mm_params",
                          help="Output directory for MM XML files (default: mm_params).")
    mm_group.add_argument("--mm-pbc", action="store_true", default=False,
                          help="Use PME (default: NoCutoff).")
    mm_group.add_argument("--mm-cutoff", type=float, default=1.0,
                          help="Nonbonded cutoff in nm (default: 1.0).")
    mm_group.add_argument("--mm-no-shake", action="store_true", default=False,
                          help="H-bond constraints.")

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
    out_group.add_argument("--log-file", type=str, default=None,
                           help="CSV log file (default: stdout only).")
    out_group.add_argument("--log-interval", type=int, default=10,
                           help="Logging interval (default: 10).")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # === System ===
    atoms: Atoms = read(args.file, format=args.format)

    # resolve QM selection — auto-detect from PDB if not given
    selection = args.selection
    if selection is None:
        selection = auto_selection(args.file, args.qm_pdb)
        print(f"Auto-detected QM selection ({len(selection)} atoms): "
              f"{selection[:10]}{'...' if len(selection) > 10 else ''}")

    # prepare MM XML files — one call per calculator
    mm_xml_qm = prepare(args.qm_pdb, args.force_fields,
                        output_dir=os.path.join(args.mm_dir, "qm"),
                        pbc=False, cutoff=args.mm_cutoff,
                        shake_h=not args.mm_no_shake)
    mm_xml_full = prepare(args.file, args.force_fields,
                          output_dir=os.path.join(args.mm_dir, "full"),
                          pbc=args.mm_pbc, cutoff=args.mm_cutoff,
                          shake_h=not args.mm_no_shake)

    # === Calculators ===
    qmcalc = QMCalculator(pdb_file=args.qm_pdb,
                          model_path=args.qm_model,
                          precision=args.qm_precision,
                          device=args.qm_device,
                          batch_size=args.qm_batch_size,
                          system_xml=mm_xml_qm.system_xml)

    mmcalc1 = MMCalculator(mm_xml_qm)
    mmcalc2 = MMCalculator(mm_xml_full)

    qmmm = QMMM(
        selection=selection,
        qmcalc=qmcalc,
        mmcalc1=mmcalc1,
        mmcalc2=mmcalc2,
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
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w")
        log_file.write("step,time_s,epot_ev,ekin_ev,temp_k\n")

    if args.traj:
        from ase.io.trajectory import TrajectoryWriter
        traj = TrajectoryWriter(args.traj, mode="w")
        dyn.attach(traj.write, interval=args.traj_interval, atoms=atoms)

    t0 = time.time()
    header = f"{'Step':>8s} {'Time/s':>10s} {'Epot/eV':>14s} {'Ekin/eV':>14s} {'T/K':>10s}"
    print(header)

    def log():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        t = time.time() - t0
        line = f"{dyn.nsteps:8d} {t:10.2f} {epot:14.4f} {ekin:14.4f} {temp:10.1f}"
        print(line)
        if log_file:
            log_file.write(f"{dyn.nsteps},{t:.3f},{epot:.4f},{ekin:.4f},{temp:.1f}\n")

    dyn.attach(log, interval=args.log_interval)

    # === Run ===
    print(f"Running {args.nsteps} steps of Langevin dynamics...")
    dyn.run(args.nsteps)
    if log_file:
        log_file.close()
    print("Done.")


if __name__ == "__main__":
    main()
