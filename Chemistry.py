from Graphs.UndirectedGraph import Node, Link, WeightedLinksUndirectedGraph

atoms = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
         "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "ge", "As", "Se", "Br",
         "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
         "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
         "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
         "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
         "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh"]

class Atom(Node):
    ID = 0

    def __init__(self, protons: int, neutrons: int, electrons: int):
        super().__init__(Atom.ID)
        Atom.ID += 1
        self.__protons = (protons - 1) % 116 + 1
        self.__name = atoms[self.protons - 1]
        self.__neutrons = abs(neutrons)
        self.__electrons = abs(electrons)
        self.__electronic_configuration = None

    @property
    def name(self):
        return self.__name

    @property
    def protons(self):
        return self.__protons

    @property
    def neutrons(self):
        return self.__neutrons

    @property
    def electrons(self):
        return self.__electrons

    @property
    def charge(self):
        return self.protons - self.electrons

    @property
    def radioactive(self):
        return (p := self.protons) in {43, 61, 84} or p > 86

    @property
    def state(self):
        if (p := self.protons) in {35, 80}:
            return "Liquid"
        if p < 3 or p in {7, 8, 9, 10, 17, 18, 36, 54, 86}:
            return "Gas"
        return "Solid"

    @property
    def artificially_acquired(self):
        return (p := self.protons) in {43, 61} or p > 92

    def similar(self, other):
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
