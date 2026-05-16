"""xTB QM/力场 calculator，为对比验证 MACE 模型的 QM 计算提供基准。"""

import os

from ase.calculators.calculator import Calculator


class XTBCalculator(Calculator):
    """xTB calculator (semi-empirical tight-binding / force field).

    需要安装 xtb-python::

        conda install -c conda-forge xtb-python

    Parameters
    ----------
    method : str
        GFN2-xTB (默认), GFN1-xTB, GFN0-xTB, 或 GFN-FF.
    charge : int
        QM 区域总电荷 (默认 0).
    uhf : int
        非成对电子数 (默认 0, 即闭壳层).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, method: str = "GFN2-xTB", charge: int = 0,
                 uhf: int = 0, **kwargs):
        super().__init__(**kwargs)
        from xtb.ase.calculator import XTB
        self._calc = XTB(method=method, charge=charge, uhf=uhf)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        pbc = atoms.pbc.copy()
        atoms.pbc = False
        saved = os.dup(1), os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            self._calc.calculate(atoms, properties, system_changes)
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            atoms.pbc = pbc
        self.results = self._calc.results
