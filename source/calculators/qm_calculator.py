import os
import sys

from ase.calculators.calculator import Calculator

# ensure lib/ is on the search path for pybind11 modules
_libdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib")
if _libdir not in sys.path:
    sys.path.insert(0, _libdir)

try:
    import libeegmfcc_solver
    _HAS_CXX = True
except ImportError as e:
    _HAS_CXX = False
    _import_error = f"{e}  (libdir={_libdir})"

from .extract_nb import extract_nb_params


class QMCalculator(Calculator):
    """QM calculator wrapping a C++ pybind11 module (``libeegmfcc_solver.EEGMFCCSolver``).

    Parameters
    ----------
    pdb_file : str
        PDB file for the QM region (topology + fragmentation).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, pdb_file: str, model_path: str,
                 precision: str = "fp32", device: str = "cpu",
                 batch_size: int = 64, system_xml: str = "", **kwargs):
        super().__init__(**kwargs)
        if not _HAS_CXX:
            raise RuntimeError(
                f"C++ module 'libeegmfcc_solver' not found: {_import_error}"
            )
        nb = extract_nb_params(system_xml) if system_xml else {}
        self._solver = libeegmfcc_solver.EEGMFCCSolver(
            pdb_file, model_path, precision, device, batch_size,
            nb.get("charges", []),
            nb.get("sigma", []),
            nb.get("epsilon", []),
            nb.get("exceptions", []))

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self._solver.compute(atoms.positions)
        self.results["energy"] = energy
        self.results["forces"] = forces
