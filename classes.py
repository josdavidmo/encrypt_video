from abc import ABCMeta
from abc import abstractmethod
from random import randint
import cv2
import numpy as np


class Attractor:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.x_0 = None
        self.y_0 = None
        self.z_0 = None

    def __str__(self):
        return "(%s, %s, %s)" % (self.x_0, self.y_0, self.z_0)

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
        Attractor.__init__(self)
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
        if self.attractor.x_0 and self.attractor.y_0 and self.attractor.z_0:
            x = self.attractor.x_0
            y = self.attractor.y_0
            z = self.attractor.z_0
        else:
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
        self.attractor.x_0 = sequence_i[0]
        self.attractor.y_0 = sequence_i[1]
        self.attractor.z_0 = sequence_i[2]
        return np.array(sequence)

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
        if np.array_equal(sequence_master, sequence_slave):
            self.attractor.x_0 = sequence_slave[0]
            self.attractor.y_0 = sequence_slave[1]
            self.attractor.z_0 = sequence_slave[2]
            return True
        return False

    def permute(self, matrix, sequence_height, sequence_width, code):
        b, g, r = cv2.split(matrix)
        if code == 1:
            b, g, r = [np.transpose(b), np.transpose(g), np.transpose(r)]
            self.make_roll(b, g, r, sequence_width, code)
            b, g, r = [np.transpose(b), np.transpose(g), np.transpose(r)]
            self.make_roll(b, g, r, sequence_height, code)
            return cv2.merge([b, g, r])
        else:
            self.make_roll(b, g, r, sequence_height, code)
            b, g, r = [np.transpose(b), np.transpose(g), np.transpose(r)]
            self.make_roll(b, g, r, sequence_width, code)
            b, g, r = [np.transpose(b), np.transpose(g), np.transpose(r)]
            return cv2.merge([b, g, r])

    def make_roll(self, b, g, r, sequence, direction):
        for i in range(len(sequence)):
            b[i] = np.roll(b[i], int(sequence[i]) * direction, axis=0)
            g[i] = np.roll(g[i], int(sequence[i]) * direction, axis=0)
            r[i] = np.roll(r[i], int(sequence[i]) * direction, axis=0)

    def difusion(self, matrix, sequence_x, sequence_y, sequence_z, code):
        matrix[:, :, 0] = (
            matrix[:, :, 0] + (code * np.repeat(sequence_x[np.newaxis].T, len(matrix[0]), axis=1))) % 256
        matrix[:, :, 1] = (
            matrix[:, :, 1] + (code * np.repeat(sequence_y[np.newaxis].T, len(matrix[0]), axis=1))) % 256
        matrix[:, :, 2] = (
            matrix[:, :, 2] + (code * np.repeat(sequence_z[np.newaxis].T, len(matrix[0]), axis=1))) % 256
        return matrix

    def encrypt(self, img):
        h = 0.01
        length = (len(img) + len(img[0])) * h
        sequence = np.rint(self.get_sequence(length, h) * 100)
        sequence_x = sequence[0:len(img)][:, 0]
        sequence_y = sequence[0:len(img)][:, 1]
        sequence_z = sequence[0:len(img)][:, 2]
        sequence_x_width = sequence[len(img) + 1:][:, 0]
        img = self.difusion(img, sequence_x, sequence_y, sequence_z, 1)
        img = self.permute(img, sequence_x, sequence_x_width, 1)
        return img

    def decrypt(self, img):
        h = 0.01
        length = (len(img) + len(img[0])) * h
        sequence = np.rint(self.get_sequence(length, h) * 100)
        sequence_x = sequence[0:len(img)][:, 0]
        sequence_y = sequence[0:len(img)][:, 1]
        sequence_z = sequence[0:len(img)][:, 2]
        sequence_x_width = sequence[len(img) + 1:][:, 0]
        img = self.permute(img, sequence_x, sequence_x_width, -1)
        img = self.difusion(img, sequence_x, sequence_y, sequence_z, -1)
        return img
