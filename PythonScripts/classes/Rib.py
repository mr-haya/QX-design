from .Airfoil import Airfoil


class Rib:
    def __init__(
        self,
        rib_number,
        y,
        chord,
        alpha,
        dihedral,
        center,
        foilname,
        z_hole_raidus,
        x_hole_radius,
    ):
        self.number = rib_number
        self.y = y
        self.chord = chord
        self.alpha = alpha
        self.dihedral = dihedral
        self.center = center
        self.foil = Airfoil(foilname)
        self.z_hole_raidus = z_hole_raidus
        self.x_hole_radius = x_hole_radius

    def thickness_margin(self):
        thickness = self.foil.thickness * self.chord
        hole_diameter = self.z_hole_raidus * 2
        return thickness - hole_diameter
