
from mesh.generators import generatePoints2dRectangle
from mesh.domain import Mesh2D
from mesh.ref_triangle import P1Triangle
from solver.solver import FEMSolver2D
from physics.material import Material
import numpy as np



h = 0.05

grid_points = generatePoints2dRectangle(-1, 1, -1, 1, h)



mesh = Mesh2D(grid_points)
mesh.plot()

def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def cFunction(x, y):
    return 1


material = Material(cFunction)
solver = FEMSolver2D(mesh, material, f, P1Triangle)

solver.assemble()
u = solver.solve()
solver.plot2d()
solver.plot3d()

