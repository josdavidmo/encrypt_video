import numpy as np
from abc import ABCMeta, abstractmethod
from random import randint

class Attractor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.x_0 = 0
        self.y_0 = 0
        self.z_0 = 0

    @abstractmethod
    def get_x(self, x, y, z, t): pass

    @abstractmethod
    def get_y(self, x, y, z, t): pass

    @abstractmethod
    def get_z(self, x, y, z, t): pass

    @abstractmethod
    def get_domain_x(self, x, y, z, t): pass

    @abstractmethod
    def get_domain_y(self, x, y, z, t): pass

    @abstractmethod
    def get_domain_z(self, x, y, z, t): pass


class Lorenz(Attractor):
    def __init__(self):
        self.a = 10
        self.b = 28
        self.c = 8 / 3

    def get_x(self, x, y, z, t):
        return self.a * (y - x)

    def get_y(self, x, y, z, t):
        return x * (self.b - z) - y

    def get_z(self, x, y, z, t):
        return x * y - self.c * z

    def get_domain_x(self):
        return randint(-10, 10)

    def get_domain_y(self):
        return randint(-15, 15)

    def get_domain_z(self):
        return randint(0, 40)

class RungeKutta4:

    def __init__(self, attractor):
        self.h = 0.01
        self.attractor = attractor

    def solve(self, x, y, z, t):
        k1 = self.h * self.attractor.get_x(x, y, z, t)
        l1 = self.h * self.attractor.get_y(x, y, z, t)
        m1 = self.h * self.attractor.get_z(x, y, z, t)
        k2 = self.h * self.attractor.get_x(x + k1 / 2, y + l1 / 2, z + m1 / 2, t + self.h / 2)
        l2 = self.h * self.attractor.get_y(x + k1 / 2, y + l1 / 2, z + m1 / 2, t + self.h / 2)
        m2 = self.h * self.attractor.get_z(x + k1 / 2, y + l1 / 2, z + m1 / 2, t + self.h / 2)
        k3 = self.h * self.attractor.get_x(x + k2 / 2, y + l2 / 2, z + m2 / 2, t + self.h / 2)
        l3 = self.h * self.attractor.get_y(x + k2 / 2, y + l2 / 2, z + m2 / 2, t + self.h / 2)
        m3 = self.h * self.attractor.get_z(x + k2 / 2, y + l2 / 2, z + m2 / 2, t + self.h / 2)
        k4 = self.h * self.attractor.get_x(x + k3, y + l3, z + m3, t + self.h)
        l4 = self.h * self.attractor.get_y(x + k3, y + l3, z + m3, t + self.h)
        m4 = self.h * self.attractor.get_z(x + k3, y + l3, z + m3, t + self.h)
        xr = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yr = (l1 + 2 * l2 + 2 * l3 + l4) / 6
        zr = (m1 + 2 * m2 + 2 * m3 + m4) / 6
        return np.array([xr, yr, zr])

class Protocolo:

    def __init__(self, attractor):
        self.attractor = attractor
        self.rj4 = RungeKutta4(attractor)

    def send_sequence(self):
        sequence = []
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        sequence = np.array([x, y, z])
        t_f = 500
        t_i = 0
        h = 0.001
        t = t_i
        while(t < t_f):
            sequence_i += self.rj4.solve(x, y, z, t)
            sequence.append(sequence_i)
            t += h
        return sequence

    def secuencia(self, len, componente):
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        secuencia = []
        i = 0
        t = 0
        h = 0.1
        while (i < len):
            caos = self.rj4.solve(x, y, z, t)
            x += caos[0]
            y += caos[1]
            z += caos[2]
            aux = int(caos[componente] % len)
            secuencia.append(aux)
            i += 1
            t += h
        return secuencia
