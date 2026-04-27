import numpy as np
from ase.calculators.calculator import Calculator
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
        pos_nm = atoms.positions / 10.0
        self._context.setPositions(pos_nm)
        state = self._context.getState(getEnergy=True, getForces=True)

        e_raw = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        f_raw = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)

        # kJ/mol → eV, kJ/mol/nm → eV/Å
        ev_per_kjmol = 1.0 / 96.485
        self.results["energy"] = float(e_raw * ev_per_kjmol)
        self.results["forces"] = f_raw * ev_per_kjmol * 0.1
