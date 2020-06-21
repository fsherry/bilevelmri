import numbers

class WaveletCoefficients:
    def __init__(self, coefs):
        self.coefs = coefs

    def __getitem__(self, key):
        if key == 0 or key == 1:
            return self.coefs[key]
        else:
            raise KeyError()

    def __setitem(self, key, x):
        if key == 0 or key == 1:
            self.coefs[key] = x
        else:
            raise KeyError()

    def __add__(self, b):
        if isinstance(b, WaveletCoefficients):
            new_coefs_l = self.coefs[0] + b.coefs[0]
            new_coefs_h = []
            for self_coefs, b_coefs in zip(self.coefs[1], b.coefs[1]):
                new_coefs_h.append(self_coefs + b_coefs)
            new_coefs = (new_coefs_l, new_coefs_h)
            return WaveletCoefficients(new_coefs)
        else:
            return NotImplemented

    def __sub__(self, b):
        if isinstance(b, WaveletCoefficients):
            new_coefs_l = self.coefs[0] - b.coefs[0]
            new_coefs_h = []
            for self_coefs, b_coefs in zip(self.coefs[1], b.coefs[1]):
                new_coefs_h.append(self_coefs - b_coefs)
            new_coefs = (new_coefs_l, new_coefs_h)
            return WaveletCoefficients(new_coefs)
        else:
            return NotImplemented

    def __truediv__(self, c):
        if isinstance(c, numbers.Real):
            new_coefs_l = self.coefs[0] / c
            new_coefs_h = []
            for self_coefs in self.coefs[1]:
                new_coefs_h.append(self_coefs / c)
            new_coefs = (new_coefs_l, new_coefs_h)
            return WaveletCoefficients(new_coefs)
        if isinstance(c, WaveletCoefficients):
            new_coefs_l = self.coefs[0] / c.coefs[0]
            new_coefs_h = []
            for self_coefs, c_coefs in zip(self.coefs[1], c.coefs[1]):
                new_coefs_h.append(self_coefs / c_coefs)
            new_coefs = (new_coefs_l, new_coefs_h)
            return new_coefs
        else:
            return NotImplemented

    def __mul__(self, c):
        if isinstance(c, numbers.Real):
            new_coefs_l = self.coefs[0] * c
            new_coefs_h = []
            for self_coefs in self.coefs[1]:
                new_coefs_h.append(self_coefs * c)
            new_coefs = (new_coefs_l, new_coefs_h)
            return WaveletCoefficients(new_coefs)
        elif isinstance(c, WaveletCoefficients):
            new_coefs_l = self.coefs[0] * c.coefs[0]
            new_coefs_h = []
            for self_coefs, c_coefs in zip(self.coefs[1], c.coefs[1]):
                new_coefs_h.append(self_coefs * c_coefs)
            new_coefs = (new_coefs_l, new_coefs_h)
        else:
            return NotImplemented

    def __rmul__(self, c):
        return self.__mul__(c)
