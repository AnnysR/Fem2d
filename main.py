
from mesh.generators import generatePoints2dRectangle, getPointsFromMatFile
from mesh.domain import Mesh2D
from mesh.ref_triangle import P1Triangle
from solver.solver import FEMSolver2D
from physics.material import Material
import numpy as np



def solve_poisson_2d(mesh, f, cFunction):
    material = Material(cFunction)
    solver = FEMSolver2D(mesh, material, f, P1Triangle)

    solver.assemble()
    u = solver.solve()
    return solver, u
    

def f(x, y):
    return (8 * np.pi**2) *np.cos(2 * np.pi * y) *(-np.pi*x+np.sin(2 * np.pi * x)) + exact_solution(x, y)
def cFunction(x, y):
    return 1

def exact_solution(x, y):
    return 2 * np.pi * x * np.cos(2 * np.pi * y) - np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

h = 0.05


grid_points = generatePoints2dRectangle(0, 1, 0, 1, h)
mesh = Mesh2D(points=grid_points)


coords , elements, edges = getPointsFromMatFile("io/i/meshS09_2.mat",0)
mesh.init_from_mat(coords, elements, edges)

print(f"Generated mesh with {mesh.points.shape[0]} points and {mesh.elements.shape[0]} elements.")
mesh.plot()
solver, u = solve_poisson_2d(mesh, f, cFunction)
max_error = solver.compute_max_error(exact_solution)

solver.plot2d()
solver.plotError2d(exact_solution)
   
   


