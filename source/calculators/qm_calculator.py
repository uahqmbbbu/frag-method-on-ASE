from ase.calculators.calculator import Calculator

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
                 batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        if not _HAS_CXX:
            raise RuntimeError(
                "C++ module 'libeegmfcc_solver' not found. "
                "Build it first:\n"
                "  cd source/calculators/cxx\n"
                "  cmake -B build && cmake --build build\n"
                "  cp build/libeegmfcc_solver*.so ../\n"
            )
        self._solver = libeegmfcc_solver.EEGMFCCSolver(
            pdb_file, model_path, precision, device, batch_size)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self._solver.compute(atoms.numbers, atoms.positions)
        self.results["energy"] = energy
        self.results["forces"] = forces
