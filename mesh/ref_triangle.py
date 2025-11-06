from mesh.domain import Mesh2D
from physics.material import Material

import numpy as np


class Triangle:
    def __init__(self, node_indices, material: Material, mesh: Mesh2D):
        self.nodes = node_indices
        self.material = material
        self.mesh = mesh
        self.area = self.compute_area()
        
    def compute_area(self):
        raise NotImplementedError("Implement in subclass")

    def stiffness_matrix(self):
        raise NotImplementedError("Implement in subclass")
    
    def load_vector(self, f):
        raise NotImplementedError("Implement in subclass")
        

class P1Triangle(Triangle):
    
    def compute_area(self):
        p = self.mesh.points
        x1, y1 = p[self.nodes[0]]
        x2, y2 = p[self.nodes[1]]
        x3, y3 = p[self.nodes[2]]
        A = 0.5 * np.linalg.det(np.array([[1,x1,y1],
                                         [1,x2,y2],
                                         [1,x3,y3]]))
        return A
    
    def stiffness_matrix(self):
        
        p = self.mesh.points
        x1, y1 = p[self.nodes[0]]
        x2, y2 = p[self.nodes[1]]
        x3, y3 = p[self.nodes[2]]


        # grad of linear shape functions
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c_ = np.array([x3 - x2, x1 - x3, x2 - x1])

        # Coefficient c(x,y) center
        xc, yc = (x1 + x2 + x3)/3, (y1 + y2 + y3)/3
        c_val = self.material.get_c(xc, yc)

        Ke = (c_val / (4*self.area)) * (np.outer(b,b) + np.outer(c_, c_))
        return Ke
    
    def load_vector(self, f):
        p = self.mesh.points
        x1, y1 = p[self.nodes[0]]
        x2, y2 = p[self.nodes[1]]
        x3, y3 = p[self.nodes[2]]

        # load vector (centre quadrature)
        xc, yc = (x1 + x2 + x3)/3, (y1 + y2 + y3)/3
        fe = np.ones(3) * f(xc, yc) * self.area / 3
        return fe
    
    
    