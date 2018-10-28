import numpy as np


class Matrix(object):
    def __init__(self, data=None):
        self.data = np.array(data)

    def shape(self):
        return self.data.shape

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        return Matrix(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data - other.data)
        return Matrix(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            shape_1 = self.shape()
            shape_2 = other.shape()
            assert shape_1[1] == shape_2[0]
            result = np.zeros((shape_1[0], shape_2[1]), dtype=np.float)

            for i in range(shape_1[0]):
                for j in range(shape_2[1]):
                    result[i][j] = self.row(i).dot_product(other.col(j))
            return Matrix(result)

        return Matrix(self.data * other)

    def __truediv__(self, other):
        return Matrix(self.data / other)

    def T(self):
        return Matrix(self.data.T)

    def col(self, i):
        return Vector(self.data[:, i])

    def row(self, i):
        return Vector(self.data[i])

    def is_squared(self):
        shape = self.shape()
        return shape[0] == shape[1]

    def is_symmetrical(self):
        h, w = self.shape()
        for i in range(h):
            for j in range(w):
                if self.data[i][j] != self.data[j][i]:
                    return False
        return True

    def minor(self, row, column):
        return Matrix(
            [np.hstack([row[:column], row[column + 1:]]) for row in np.vstack([self.data[:row], self.data[row + 1:]])])

    def det(self):
        N, _ = self.shape()
        if N == 1:
            return self.data[0][0]
        result = 0
        factor = 1
        for i in range(N):
            result += factor * self.data[0][i] * self.minor(0, i).det()
            factor *= -1
        return result

    def inv(self):
        N = self.shape()[0]
        cofactors = np.zeros((N, N))
        for row in range(N):
            for column in range(N):
                cofactors[row][column] = (-1) ** (row + column) * self.minor(row, column).det()
        adj = Matrix(cofactors).T()
        return adj / self.det()

    def diag(self):
        N = self.shape()[0]
        return Vector(self.data[np.arange(N), np.arange(N)])

    def sum(self):
        return self.data.sum()

    def min(self):
        return np.min(self.data)

    def max(self):
        return np.max(self.data)

    def abs(self):
        return Matrix(np.abs(self.data))

    def __str__(self):
        s = ''
        for elem in self.data:
            s += str(list(elem)) + '\n'
        return s

    def __repr__(self):
        return str(self)


class Vector(Matrix):
    def __init__(self, data=None):
        if isinstance(data, Matrix):
            super().__init__(data.data.reshape((-1, 1)))
        else:
            super().__init__(np.array(data).reshape((-1, 1)))

    def __mul__(self, other):
        if isinstance(self, Vector):
            return Vector(self.data * other.data)
        return Vector(self.data * other)

    def abs(self):
        return Vector(np.abs(self.data))

    def dot_product(self, other):
        return (self * other).sum()

    def norm(self):
        return np.sqrt(self.dot_product(self))
