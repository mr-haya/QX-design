class Spar:
    def __init__(self, name, start, end, laminate, ellipticity, taper_ratio):
        self.name = name
        self.start = start
        self.end = end
        self.laminate = laminate
        self.ellipticity = ellipticity
        self.taper_ratio = taper_ratio


class WingSpar:  # 任意位置での断面と剛性率を返す
    def __init__(self, spar0, spar1, spar2, spar3):
        self.spar0 = spar0
        self.spar1 = spar1
        self.spar2 = spar2
        self.spar3 = spar3
