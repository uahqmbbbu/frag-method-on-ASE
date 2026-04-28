from ase.calculators.calculator import Calculator
from ase.calculators.qmmm import SimpleQMMM


class QMMM(SimpleQMMM):
    """
    Simplified QMMM calculator. Runs QM and MM calculations sequentially.

    Parameters
    ----------
        selection: list[int]
            Indices of atoms in the QM region.
        qmcalc: Calculator
            QM calculator.
        mmcalc1: Calculator
            MM calculator applied to the QM region (for subtraction).
        mmcalc2: Calculator
            MM calculator applied to the full system.
        vacuum: bool
            If True, remove net translation of QM forces.
        center: list[float] | None
            Center position for the QM region when vacuum=True.
    """

    def get_qmcalc_results(self, properties, system_changes):
        self.qmcalc.calculate(self.qmatoms, properties, system_changes)
        return (
            self.qmcalc.results["energy"],
            self.qmcalc.results["forces"],
        )

    def get_mmcalc1_results(self, properties, system_changes):
        self.mmcalc1.calculate(self.qmatoms, properties, system_changes)
        return (
            self.mmcalc1.results["energy"],
            self.mmcalc1.results["forces"],
        )

    def get_mmcalc2_results(self, properties, system_changes):
        self.mmcalc2.calculate(self.atoms, properties, system_changes)
        return (
            self.mmcalc2.results["energy"],
            self.mmcalc2.results["forces"],
        )

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.qmatoms is None:
            self.qmatoms = atoms[self.selection]
        self.qmatoms.positions = atoms.positions[self.selection]
        if self.vacuum:
            self.qmatoms.positions += self.center - self.qmatoms.positions.mean(axis=0)

        qm_e, qm_f = self.get_qmcalc_results(properties, system_changes)
        mm1_e, mm1_f = self.get_mmcalc1_results(properties, system_changes)
        mm2_e, mm2_f = self.get_mmcalc2_results(properties, system_changes)

        energy = mm2_e + qm_e - mm1_e
        forces = mm2_f.copy()

        if self.vacuum:
            qm_f -= qm_f.mean(axis=0)

        forces[self.selection] += qm_f - mm1_f

        self.results["energy"] = energy
        self.results["forces"] = forces
