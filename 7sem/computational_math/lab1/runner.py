from sim_task import SIMTask
from objects import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--max-iter', dest='max_iterations', type=int, help='maximum number of iterations', default=100)
parser.add_argument('--tau', dest='tau', type=str, help='tau for SI method', default='gershgorin')
parser.add_argument('--verbose', dest='verbose', action='store_true')
args = parser.parse_args()

N = int(input())
eps = float(input())
f = Vector([float(input()) for _ in range(N)])
a = Matrix([[float(input()) for i in range(N)] for j in range(N)])

task = SIMTask(a, f, max_iterations=args.max_iterations, eps=eps, tau=args.tau, verbose=args.verbose)
task.run()
task.print_solution()
