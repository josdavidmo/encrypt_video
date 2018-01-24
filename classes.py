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
        self.attractor = attractor

    def solve(self, x, y, z, t, h):
        k1 = h * self.attractor.get_x(x, y, z, t)
        l1 = h * self.attractor.get_y(x, y, z, t)
        m1 = h * self.attractor.get_z(x, y, z, t)
        k2 = h * \
            self.attractor.get_x(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + h / 2)
        l2 = h * \
            self.attractor.get_y(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + h / 2)
        m2 = h * \
            self.attractor.get_z(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + h / 2)
        k3 = h * \
            self.attractor.get_x(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + h / 2)
        l3 = h * \
            self.attractor.get_y(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + h / 2)
        m3 = h * \
            self.attractor.get_z(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + h / 2)
        k4 = h * self.attractor.get_x(x + k3, y + l3, z + m3, t + h)
        l4 = h * self.attractor.get_y(x + k3, y + l3, z + m3, t + h)
        m4 = h * self.attractor.get_z(x + k3, y + l3, z + m3, t + h)
        xr = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yr = (l1 + 2 * l2 + 2 * l3 + l4) / 6
        zr = (m1 + 2 * m2 + 2 * m3 + m4) / 6
        return np.array([xr, yr, zr])


class Protocol:

    def __init__(self, attractor):
        self.attractor = attractor
        self.rk4 = RungeKutta4(attractor)

    def get_sequence(self, length, h=0.01):
        sequence = []
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        sequence_i = np.array([x, y, z])
        t_f = length
        t_i = 0
        t = t_i
        while(t < t_f):
            x, y, z = sequence_i
            sequence_i = sequence_i + self.rk4.solve(x, y, z, t, h)
            sequence.append(sequence_i)
            t += h
        self.x_0 = sequence_i[0]
        self.y_0 = sequence_i[1]
        self.z_0 = sequence_i[2]
        return sequence

    def synchronize(self, sequence, h=0.01):
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        sequence_slave = np.array([x, y, z])
        t_f = len(sequence)
        t_i = 0
        t = t_i
        for sequence_master in sequence:
            x, y, z = sequence_slave
            sequence_slave = sequence_slave + self.rk4.solve(x, y, z, t, h)
            sequence_slave[1] = sequence_master[1]
            t += h
            if np.array_equal(sequence_master, sequence_slave) and t < t_f:
                print t
                self.x_0 = sequence_slave[0]
                self.y_0 = sequence_slave[1]
                self.z_0 = sequence_slave[2]
                return True
        return False

    def encrypt(self, img):
        sequence_heigth = self.get_sequence(len(img))
        sequence_width = self.get_sequence(len(img[0]))
