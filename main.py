
from mesh.generators import generatePoints2dRectangle
from mesh.domain import Mesh2D
from mesh.ref_triangle import P1Triangle
from solver.solver import FEMSolver2D
from physics.material import Material



h = 0.05
points = generatePoints2dRectangle(0, 1, 0, 1, h)
mesh = Mesh2D(points)
mesh.plot()

def f(x, y):
    return 1.0

def c_function(x, y):
    return 1.0


material = Material(c_function)
solver = FEMSolver2D(mesh, material, f, P1Triangle)

solver.assemble()
u = solver.solve()
solver.plot2d()
solver.plot3d()

