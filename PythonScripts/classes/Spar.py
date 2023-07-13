import math
from .Laminate import Laminate


class Spar:
    def __init__(self, name, length, laminate, ellipticity, taper_ratio, zi):
        self.name = name
        self.length = length
        self.laminate = Laminate(laminate)
        self.ellipticity = ellipticity
        self.taper_ratio = taper_ratio
        self.zi = zi

        self.E = self.laminate.E_equiv
        self.G = self.laminate.G_equiv
        self.nu = self.laminate.nu_equiv

    def zixizoxo(self, x):
        if x < 0 or x > self.length:
            return 0, 0, 0, 0
        else:
            zi = (self.zi - self.taper_ratio * x) / 2
            xi = zi * self.ellipticity
            zo = zi + self.laminate.thickness
            xo = xi + self.laminate.thickness_zenshu

            return zi, xi, zo, xo

    def section_modulus(self, x):
        return calc_section_modulus(self.zixizoxo(x))


class WingSpar:  # 任意位置での断面と剛性率を返す
    def __init__(
        self, spar0, spar1_start, spar1, spar2_start, spar2, spar3_start, spar3
    ):
        self.spar0 = spar0
        self.spar1 = spar1
        self.spar2 = spar2
        self.spar3 = spar3
        self.spar1_start = spar1_start
        self.spar2_start = spar2_start
        self.spar3_start = spar3_start

    def zixizoxo(self, x):
        if x < self.spar1_start:
            return self.spar0.zixizoxo(x)
        elif x < self.spar0.length / 2:
            x_1 = x - self.spar1_start
            zi0, xi0, zo0, xo0 = self.spar0.zixizoxo(x)
            zi1, xi1, zo1, xo1 = self.spar1.zixizoxo(x_1)
            return zi0, xi0, zo1, xo1
        elif x < self.spar2_start:
            x_1 = x - self.spar1_start
            return self.spar1.zixizoxo(x_1)
        elif x < self.spar1_start + self.spar1.length:
            x_1 = x - self.spar2_start
            x_2 = x - self.spar2_start
            zi1, xi1, zo1, xo1 = self.spar1.zixizoxo(x_1)
            zi2, xi2, zo2, xo2 = self.spar2.zixizoxo(x_2)
            return zi1, xi1, zo2, xo2
        elif x < self.spar3_start:
            x_2 = x - self.spar2_start
            return self.spar2.zixizoxo(x_2)
        elif x < self.spar2_start + self.spar2.length:
            x_2 = x - self.spar2_start
            x_3 = x - self.spar3_start
            zi2, xi2, zo2, xo2 = self.spar2.zixizoxo(x_2)
            zi3, xi3, zo3, xo3 = self.spar3.zixizoxo(x_3)
            return zi2, xi2, zo3, xo3
        elif x < self.spar3_start + self.spar3.length:
            x_3 = x - self.spar3_start
            return self.spar3.zixizoxo(x_3)
        else:
            return 0, 0, 0, 0

    def section_modulus(self, x):
        return calc_section_modulus(self.zixizoxo(x))

    def E(self, x):
        if x < self.spar1_start:
            return self.spar0.E
        elif x < self.spar0.length / 2:
            return (self.spar0.E + self.spar1.E) / 2
        elif x < self.spar2_start:
            return self.spar1.E
        elif x < self.spar1_start + self.spar1.length:
            return (self.spar1.E + self.spar2.E) / 2
        elif x < self.spar3_start:
            return self.spar2.E
        elif x < self.spar2_start + self.spar2.length:
            return (self.spar2.E + self.spar3.E) / 2
        elif x < self.spar3_start + self.spar3.length:
            return self.spar3.E
        else:
            return self.spar3.E

    def G(self, x):
        if x < self.spar1_start:
            return self.spar0.G
        elif x < self.spar0.length / 2:
            return (self.spar0.G + self.spar1.G) / 2
        elif x < self.spar2_start:
            return self.spar1.G
        elif x < self.spar1_start + self.spar1.length:
            return (self.spar1.G + self.spar2.G) / 2
        elif x < self.spar3_start:
            return self.spar2.G
        elif x < self.spar2_start + self.spar2.length:
            return (self.spar2.G + self.spar3.G) / 2
        elif x < self.spar3_start + self.spar3.length:
            return self.spar3.G
        else:
            return self.spar3.G

    def nu(self, x):
        if x < self.spar1_start:
            return self.spar0.nu
        elif x < self.spar0.length / 2:
            return (self.spar0.nu + self.spar1.nu) / 2
        elif x < self.spar2_start:
            return self.spar1.nu
        elif x < self.spar1_start + self.spar1.length:
            return (self.spar1.nu + self.spar2.nu) / 2
        elif x < self.spar3_start:
            return self.spar2.nu
        elif x < self.spar2_start + self.spar2.length:
            return (self.spar2.nu + self.spar3.nu) / 2
        elif x < self.spar3_start + self.spar3.length:
            return self.spar3.nu
        else:
            return self.spar3.nu


def calc_section_modulus(a_in, b_in, a_out, b_out, num=32766):
    pi = math.pi
    dtheta = (pi / 2) / num
    a_cl = (a_in + a_out) / 2
    b_cl = (b_in + b_out) / 2
    Am = pi * a_cl * b_cl

    Ix, Iy, j, Area, Length = 0, 0, 0, 0, 0

    for i in range(num + 1):
        theta = dtheta * i

        # Calculate x, y, r for inner, outer, and center line
        x_in, y_in, r_in = calculate_coordinates(a_in, b_in, theta)
        x_out, y_out, r_out = calculate_coordinates(a_out, b_out, theta)
        x_cl, y_cl, r_cl = calculate_coordinates(a_cl, b_cl, theta)

        # Calculate phi, ds, dr, dA, t
        phi = pi / 2 if theta == 0 else math.atan(-((b_cl / a_cl) ** 2) * (x_cl / y_cl))
        ds = (
            r_cl * dtheta
            if theta == 0
            else r_cl * dtheta / math.cos(theta - pi / 2 - phi)
        )
        dr = r_out - r_in
        da = r_cl * dtheta * dr
        thickness = dr * math.cos(theta - pi / 2 - phi)

        # Calculate Ix, Iy, J, L, A
        Ix += y_cl * y_cl * da
        Iy += x_cl * x_cl * da
        j += ds / thickness
        Length += ds
        Area += da

    # Calculate J
    j = (4 * Am * Am) / (j * 4)

    # Store section factors in array
    data = [Ix * 4, Iy * 4, j, Area * 4, Am, Length * 4]

    return data


def calculate_coordinates(a, b, theta):
    x = a * math.cos(theta)
    y = b * math.sin(theta)
    r = math.sqrt(x**2 + y**2)
    return x, y, r
