"""从 OpenMM system.xml 提取非键合参数，供 C++ NonBondedSolver 使用。

将此逻辑放在 Python 层，避免 C++ 扩展直接链接 OpenMM C++ 库，
从而解决 conda OpenMM (CXX ABI=1) 与 conda/PyTorch (CXX ABI=0) 的兼容问题。
"""

from openmm import NonbondedForce, XmlSerializer
from openmm import unit


def extract_nb_params(system_xml: str):
    """从 system.xml 提取非键合参数。

    Returns
    -------
    dict
        charges (list[float]), sigma (list[float], nm),
        epsilon (list[float], kJ/mol),
        exceptions (list[list[float]])
        — 每个例外对为 [i, j, chargeProd, sigma_nm, epsilon_kJmol]。
    """
    with open(system_xml) as f:
        system = XmlSerializer.deserialize(f.read())

    nb = None
    for k in range(system.getNumForces()):
        force = system.getForce(k)
        if force.getName() == "NonbondedForce":
            nb = force
            break

    if nb is None:
        raise ValueError("NonbondedForce not found in system XML.")

    N = nb.getNumParticles()
    charges = []
    sigma = []
    epsilon = []
    for i in range(N):
        q, s, e = nb.getParticleParameters(i)
        charges.append(q.value_in_unit(unit.elementary_charge))
        sigma.append(s.value_in_unit(unit.nanometer))
        epsilon.append(e.value_in_unit(unit.kilojoule_per_mole))

    exceptions = []
    for k in range(nb.getNumExceptions()):
        i, j, qq, sig, eps = nb.getExceptionParameters(k)
        if i > j:
            i, j = j, i
        exceptions.append([
            float(i), float(j),
            qq.value_in_unit(unit.elementary_charge**2),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole),
        ])

    return {
        "charges": charges,
        "sigma": sigma,
        "epsilon": epsilon,
        "exceptions": exceptions,
    }
