from .UndirectedGraph import Node, Link, WeightedLinksUndirectedGraph

atoms = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
         "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "ge", "As", "Se", "Br",
         "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
         "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
         "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
         "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
         "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh"]
full_e_c = ["1s", "1s2", "1s2|2s", "1s2|2s2", "1s2|2s2|2p", "1s2|2s2|2p2", "1s2|2s2|2p3", "1s2|2s2|2p4", "1s2|2s2|2p5",
"1s2|2s2|2p6", "1s2|2s2|2p6|3s", "1s2|2s2|2p6|3s2", "1s2|2s2|2p6|3s2|3p", "1s2|2s2|2p6|3s2|3p2", "1s2|2s2|2p6|3s2|3p3",
            "1s2|2s2|2p6|3s2|3p4", "1s2|2s2|2p6|3s2|3p5", "1s2|2s2|2p6|3s2|3p6", "1s2|2s2|2p6|3s2|3p6|4s",
"1s2|2s2|2p6|3s2|3p6|4s2", "1s2|2s2|2p6|3s2|3p6|3d|4s2", "1s2|2s2|2p6|3s2|3p6|3d2|4s2", "1s2|2s2|2p6|3s2|3p6|3d3|4s2",
            "1s2|2s2|2p6|3s2|3p6|3d5|4s", "1s2|2s2|2p6|3s2|3p6|3d5|4s2", "1s2|2s2|2p6|3s2|3p6|3d6|4s2",
            "1s2|2s2|2p6|3s2|3p6|3d7|4s2", "1s2|2s2|2p6|3s2|3p6|3d8|4s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p3", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p4", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p5",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|5s", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|5s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d|5s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d2|5s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d4|5s", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d5|5s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d5|5s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d7|5s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d8|5s", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p3", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p4",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p5", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|6s", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|5d|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f|5d|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f3|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f4|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f5|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f6|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f7|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f7|5d|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f9|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f10|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f11|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f12|6s2",
    "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f13|6s2", "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d2|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d3|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d4|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d5|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d6|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d7|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d9|6s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p3",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p4",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p5",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|7s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|6d|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|6d2|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f2|6d|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f3|6d|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f4|6d|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f6|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f7|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f7|6d|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f9|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f10|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f11|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f12|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f13|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|7s2|7p",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d2|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d3|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d4|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d5|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d6|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d7|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d9|7s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d10|7s",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d10|7s2",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|6d10|7s2|7p",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|7s2|7p2"
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|7s2|7p3",
            "1s2|2s2|2p6|3s2|3p6|3d10|4s2|4p6|4d10|5s2|5p6|4f14|5d10|6s2|6p6|5f14|7s2|7p4"]

class Atom(Node):
    ID = 0

    def __init__(self, protons: int, neutrons: int, electrons: int):
        super().__init__(Atom.ID)
        Atom.ID += 1
        self.__protons = (protons - 1) % 116 + 1
        self.__name = atoms[self.protons - 1]
        self.__neutrons = abs(neutrons)
        self.__electrons = abs(electrons)
        self.__electronic_configuration = full_e_c[self.protons - 1]

    @property
    def name(self) -> str:
        return self.__name

    @property
    def protons(self) -> int:
        return self.__protons

    @property
    def neutrons(self) -> int:
        return self.__neutrons

    @property
    def electrons(self) -> int:
        return self.__electrons

    @property
    def charge(self) -> int:
        return self.protons - self.electrons

    @property
    def radioactive(self) -> bool:
        return (p := self.protons) in {43, 61, 84} or p > 86

    @property
    def state(self) -> str:
        if (p := self.protons) in {35, 80}:
            return "Liquid"
        if p < 3 or p in {7, 8, 9, 10, 17, 18, 36, 54, 86}:
            return "Gas"
        return "Solid"

    @property
    def artificially_acquired(self) -> bool:
        return (p := self.protons) in {43, 61} or p > 92

    def similar(self, other) -> bool:
        if isinstance(other, Atom):
            return (self.protons, self.neutrons, self.electrons) == (other.protons, other.neutrons, other.electrons)
        return False

    def __gt__(self, other):
        if isinstance(other, Atom):
            return self.protons > other.protons
        return False

    def __lt__(self, other):
        if isinstance(other, Atom):
            return self.protons < other.protons
        return False

    def __ge__(self, other):
        if isinstance(other, Atom):
            return self.protons >= other.protons
        return False

    def __le__(self, other):
        if isinstance(other, Atom):
            return self.protons <= other.protons
        return False

    def __str__(self):
        return self.name + str(self.charge)

    __repr__ = __str__


class Bond(Link):
    pass


class Molecule(WeightedLinksUndirectedGraph):
    pass
