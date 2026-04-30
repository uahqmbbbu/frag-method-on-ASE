"""Generate OpenMM XML files for an MM calculator."""

import os
from dataclasses import dataclass

from openmm import LangevinIntegrator, XmlSerializer
from openmm.app import ForceField, PDBFile, PDBxFile, PME, NoCutoff


@dataclass
class MMXMLSet:
    system_xml: str
    integrator_xml: str


def prepare(pdb_path: str,
            force_fields: list[str],
            output_dir: str = ".",
            temperature: float = 300.0,
            friction: float = 1.0,
            timestep: float = 0.001,  # ps (OpenMM unit; 0.001 ps = 1 fs)
            pbc: bool = True,
            cutoff: float = 1.0) -> MMXMLSet:
    """Generate System XML and Integrator XML from a PDB and force fields.

    Call once for each MM calculator.
    """
    os.makedirs(output_dir, exist_ok=True)

    _, ext = os.path.splitext(pdb_path)
    pdb = PDBFile(pdb_path) if ext.lower() == ".pdb" else PDBxFile(pdb_path)

    # integrator (not used for dynamics, just needed for Context creation)
    integrator = LangevinIntegrator(temperature, friction, timestep)
    integrator_xml = os.path.join(output_dir, "integrator.xml")
    with open(integrator_xml, "w") as f:
        f.write(XmlSerializer.serialize(integrator))

    # system: no constraints, flexible water — forces are complete
    ff = ForceField(*force_fields)
    method = PME if pbc else NoCutoff
    system = ff.createSystem(pdb.topology,
                             nonbondedMethod=method,
                             nonbondedCutoff=cutoff,
                             constraints=None,
                             rigidWater=False)
    system_xml = os.path.join(output_dir, "system.xml")
    with open(system_xml, "w") as f:
        f.write(XmlSerializer.serialize(system))

    return MMXMLSet(system_xml=system_xml, integrator_xml=integrator_xml)
