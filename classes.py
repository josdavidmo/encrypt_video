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
        k2 = self.h * \
            self.attractor.get_x(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + self.h / 2)
        l2 = self.h * \
            self.attractor.get_y(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + self.h / 2)
        m2 = self.h * \
            self.attractor.get_z(x + k1 / 2, y + l1 / 2,
                                 z + m1 / 2, t + self.h / 2)
        k3 = self.h * \
            self.attractor.get_x(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + self.h / 2)
        l3 = self.h * \
            self.attractor.get_y(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + self.h / 2)
        m3 = self.h * \
            self.attractor.get_z(x + k2 / 2, y + l2 / 2,
                                 z + m2 / 2, t + self.h / 2)
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

    def get_sequence(self, length, h=0.001):
        sequence = []
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        sequence_i = np.array([x, y, z])
        t_f = length
        t_i = 0
        t = t_i
        while(t < t_f):
            sequence_i += self.rj4.solve(x, y, z, t)
            sequence.append(sequence_i)
            t += h
        self.x_0 = sequence_i[0]
        self.y_0 = sequence_i[1]
        self.z_0 = sequence_i[2]
        return sequence

    def synchronize(self, sequence, h=0.001):
        x = self.attractor.get_domain_x()
        y = self.attractor.get_domain_y()
        z = self.attractor.get_domain_z()
        sequence_slave = np.array([x, y, z])
        length = len(sequence)
        t_f = len(sequence)
        t_i = 0
        t = t_i
        for sequence_master in sequence:
            sequence_slave += self.rk4.solve(xr, y, zr, t)
            sequence_slave[1] = sequence_master[1]
            t += h
            if np.array_equal(sequence_master, sequence_slave) or t > t_f:
                self.x_0 = sequence_slave[0]
                self.y_0 = sequence_slave[1]
                self.z_0 = sequence_slave[2]
                return True
        return False

    def permute(self,matrix,sequence_height,sequence_width,code):
        b,g,r = cv2.split(matrix)
        if code == 1:
            b,g,r = [np.transpose(b),np.transpose(g),np.transpose(r)]
            make_roll(b,g,r,sequence_width,code)
            b,g,r = [np.transpose(b),np.transpose(g),np.transpose(r)]
            make_roll(b,g,r,sequence_height,code)
            return cv2.merge([b,g,r])
        else:
            make_roll(b,g,r,sequence_height,code)
            b,g,r = [np.transpose(b),np.transpose(g),np.transpose(r)]
            make_roll(b,g,r,sequence_width,code)
            b,g,r = [np.transpose(b),np.transpose(g),np.transpose(r)]
            return cv2.merge([b,g,r])

    def make_roll(b,g,r,sequence,direction):
        for i in range(len(sequence)):
            b[i] = np.roll(b[i], sequence[i]*direction, axis=0)
            g[i] = np.roll(g[i], sequence[i]*direction, axis=0)
            r[i] = np.roll(r[i], sequence[i]*direction, axis=0)

    def difusion(self,matrix,sequence_x,sequence_y,sequence_z,code):
        b,g,r = cv2.split(matrix)
        b = b + np.repeat(sequence_x[np.newaxis].T,len(matrix[0]),axis=1)
        g = g + np.repeat(sequence_y[np.newaxis].T,len(matrix[0]),axis=1)
        r = r + np.repeat(sequence_z[np.newaxis].T,len(matrix[0]),axis=1)
        return cv2.merge([b,g,r])
