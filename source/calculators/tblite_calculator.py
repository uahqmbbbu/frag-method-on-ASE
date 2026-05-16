"""tblite QM calculator，纯 Python xTB 实现，无需 xtb 二进制。"""

from ase.calculators.calculator import Calculator


class TBLiteCalculator(Calculator):
    """QM calculator using tblite (light-weight tight-binding).

    需要安装 tblite::

        conda install -c conda-forge tblite-python

    Parameters
    ----------
    method : str
        GFN2-xTB (默认), GFN1-xTB, 或 GFN0-xTB.
    charge : int
        QM 区域总电荷 (默认 0).
    uhf : int
        非成对电子数 (默认 0, 即闭壳层).
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, method: str = "GFN2-xTB", charge: int = 0,
                 uhf: int = 0, **kwargs):
        super().__init__(**kwargs)
        from tblite.ase import TBLite
        self._calc = TBLite(method=method, charge=charge, uhf=uhf)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        pbc = atoms.pbc.copy()
        atoms.pbc = False
        try:
            self._calc.calculate(atoms, properties, system_changes)
        finally:
            atoms.pbc = pbc
        self.results = self._calc.results
