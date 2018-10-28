import numpy as np
from objects import *


class SIMTask(object):
    def __init__(self, A, f, max_iterations=100, eps=1e-5, tau='gershgorin', verbose=False):
        self.A = A
        self.f = f
        self.N = A.shape()[0]
        self.max_iterations = max_iterations
        self.eps = eps
        self.verbose = verbose

        self.configure_data()

        # Configuring tau
        if tau == 'gershgorin':
            self.tau = self.gershgorin_approximation()
        else:
            self.tau = float(tau)

        self.iteration = 0
        self.previous = Vector(np.ones((self.N, 1)) * 1e7)
        self.current = Vector(np.ones((self.N, 1)))

    def gershgorin_approximation(self):
        diag = self.A.diag()
        dists = []
        for i in range(self.A.shape()[0]):
            dists.append(self.A.row(i).abs().sum() - abs(self.A.data[i][i]))

        minimum = diag - Vector(dists)
        maximum = diag + Vector(dists)
        lambda_min = minimum.min()
        lambda_max = maximum.max()
        tau = 2 / (lambda_min + lambda_max)
        if self.verbose:
            print('Gershgorin tau approximation: {} (lambd_min={}, lambd_max={})'.format(tau, lambda_min, lambda_max))
        return 2 / (lambda_min + lambda_max)

    def configure_data(self):
        if not self.A.is_squared():
            raise ValueError("A must be symmetrical")
        if not self.A.is_symmetrical():
            B = self.A.T() * self.A
            F = self.A.T() * self.f
            self.A, self.f = B, F

    def run(self):
        while self.iteration < self.max_iterations and self.current_eps() > self.eps:
            self.make_iteration()

    def current_eps(self):
        return Vector(self.current - self.previous).norm()

    def make_iteration(self):
        self.iteration += 1
        error = Vector(self.A * self.current) - self.f
        self.previous, self.current = self.current, Vector(error * (-self.tau) + self.current)
        if self.verbose:
            print("iteration {}, solution:\n{}".format(self.iteration, self.current))

    def print_solution(self):
        print("solution (after {} iterations):\n{}".format(self.iteration, self.current))
