import numpy as np
from ase.calculators.calculator import Calculator
from ase.units import kJ, mol, nm
from openmm import Context, Platform, XmlSerializer, unit


class MMCalculator(Calculator):
    """Full-force-field MM calculator via Python OpenMM API.

    Parameters
    ----------
    xml_set : MMXMLSet
        Paths to system.xml and integrator.xml.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, xml_set, **kwargs):
        super().__init__(**kwargs)

        with open(xml_set.system_xml) as f:
            system = XmlSerializer.deserialize(f.read())
        with open(xml_set.integrator_xml) as f:
            integrator = XmlSerializer.deserialize(f.read())

        platform = Platform.getPlatformByName("CPU")
        self._context = Context(system, integrator, platform)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        # Å → nm
        self._context.setPositions(atoms.positions / 10.0)
        state = self._context.getState(getEnergy=True, getForces=True)

        e_raw = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        f_raw = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)

        # kJ/mol → eV,   kJ/mol/nm → eV/Å
        self.results["energy"] = float(e_raw * (kJ / mol))
        self.results["forces"] = f_raw * (kJ / mol / nm)

        from .force_debug import log_forces
        log_forces("MMCalc", self.results["forces"], atoms.numbers)
