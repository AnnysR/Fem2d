from mesh.domain import Mesh2D
from physics.material import Material

import numpy as np


class Triangle:
    def __init__(self, node_indices, material: Material, mesh: Mesh2D):
        self.nodes = node_indices
        self.material = material
        self.mesh = mesh
        self.J = self.computeJacobian()
        self.area = self.computeArea()
        self.center = self.calculateCenter()

    def computeArea(self):
        raise NotImplementedError("Implement in subclass")

    def stiffnessMatrix(self):
        raise NotImplementedError("Implement in subclass")

    def loadVector(self, f):
        raise NotImplementedError("Implement in subclass")

    def computeJacobian(self):
        raise NotImplementedError("Implement in subclass")

    def calculateCenter(self):
        p = self.mesh.points
        x_coords = [p[i][0] for i in self.nodes]
        y_coords = [p[i][1] for i in self.nodes]
        center_x = sum(x_coords) / len(self.nodes)
        center_y = sum(y_coords) / len(self.nodes)
        return np.array([center_x, center_y])


class P1Triangle(Triangle):
    def computeArea(self):
        A = 0.5 * np.linalg.det(self.J)
        return A

    def computeJacobian(self):
        p = self.mesh.points
        x1, y1 = p[self.nodes[0]]
        x2, y2 = p[self.nodes[1]]
        x3, y3 = p[self.nodes[2]]
        J = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        return J


    def stiffnessMatrix(self):
        # reference gradients (3×2)
        dN_ref = np.array([[-1, -1], [1, 0], [0, 1]])

        # inverse Jacobian
        invJ = np.linalg.inv(self.J)

        # physical gradients (3×2)
        dN = dN_ref @ invJ

        c = self.material.cFunction(*self.center)

        # stiffness matrix (3×3)
        Ke = self.area * c * (dN @ dN.T)
    

        return Ke

    def loadVector(self, f):
        xc, yc = self.center
        fe = np.ones(3) * f(xc, yc) * self.area / 3
        return fe

    def triangleJacobian(self):
        # jacobian of the transformation from reference triangle to physical triangle
        p = self.mesh.points
        x1, y1 = p[self.nodes[0]]
        x2, y2 = p[self.nodes[1]]
        x3, y3 = p[self.nodes[2]]
        J = np.array([[x2 - x1, x3 - x1], [y2 - y1, y3 - y1]])
        return J
