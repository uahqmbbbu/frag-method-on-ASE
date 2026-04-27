import os
import sys

from ase.calculators.calculator import Calculator

# ensure lib/ is on the search path for pybind11 modules
_libdir = os.path.join(os.path.dirname(__file__), "..", "..", "lib")
if _libdir not in sys.path:
    sys.path.insert(0, _libdir)

try:
    import libeegmfcc_solver
    _HAS_CXX = True
except ImportError:
    _HAS_CXX = False


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
                "C++ module 'libeegmfcc_solver' not found."
                "  Build it: cd source/cxx && bash compile.sh"
            )
        self._solver = libeegmfcc_solver.EEGMFCCSolver(
            pdb_file, model_path, precision, device, batch_size, system_xml)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self._solver.compute(atoms.positions)
        self.results["energy"] = energy
        self.results["forces"] = forces
