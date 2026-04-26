from ase.calculators.calculator import Calculator


class MMCalculator(Calculator):
    """MM calculator stub.

    Implement calculate() to call your C++ MM solver.

    Input:
        atoms.numbers:   atomic numbers
        atoms.positions: atomic positions

    Output (set on self.results):
        "energy": total energy (float)
        "forces": per-atom forces, shape (n_atoms, 3)
    """

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        raise NotImplementedError("MMCalculator.calculate must be implemented.")
