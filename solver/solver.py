import numpy as np
from mesh.domain import Mesh2D
from physics.material import Material

import matplotlib.pyplot as plt


class FEMSolver2D:
    def __init__(self, mesh: Mesh2D, material: Material, f, triangle_class):
        self.mesh = mesh
        self.material = material
        self.f = f
        self.triangle_class = triangle_class
        self.K = None
        self.F = None
        self.solution = None

    def assemble(self):
        GlobalStiffness = np.zeros(
            (self.mesh.points.shape[0], self.mesh.points.shape[0])
        )
        GlobalLoad = np.zeros(self.mesh.points.shape[0])

        for elem_nodes in self.mesh.elements:
            triangle = self.triangle_class(elem_nodes, self.material, self.mesh)
            Ke = triangle.stiffnessMatrix()
            fe = triangle.loadVector(self.f)

            for i_local, i_global in enumerate(elem_nodes):
                GlobalLoad[i_global] += fe[i_local]
                for j_local, j_global in enumerate(elem_nodes):
                    GlobalStiffness[i_global, j_global] += Ke[i_local, j_local]

        self.K = GlobalStiffness
        self.F = GlobalLoad

    def applyDirichletBC(self, K, F, boundary_nodes):
        n_nodes = self.mesh.points.shape[0]
        mask = np.ones(n_nodes, dtype=bool)
        mask[boundary_nodes] = False
        K_reduced = K[mask][:, mask]
        F_reduced = F[mask]
        return K_reduced, F_reduced, mask

    def solve(self):
        if self.K is None or self.F is None:
            raise ValueError("System not assembled. Call assemble() before solve().")

        # Apply Dirichlet BC as before
        K_bc, F_bc, mask = self.applyDirichletBC(
            self.K, self.F, self.mesh.boundary_nodes
        )
        u_reduced = np.linalg.solve(K_bc, F_bc)
        # Reconstruct full solution
        u_full = np.zeros(self.mesh.points.shape[0])
        u_full[mask] = u_reduced

        self.solution = u_full
        return u_full

    def plot2d(self):
        if self.solution is None:
            raise ValueError("No solution available. Please run solve() first.")

        plt.tricontourf(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.elements,
            self.solution,
            levels=14,
            cmap="viridis",
        )
        plt.triplot(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.elements,
            color="black",
            alpha=0.3,
            linewidth=0.5,
        )
        plt.colorbar(label="Solution u")
        plt.title("FEM Solution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()

    def plot3d(self):
        if self.solution is None:
            raise ValueError("No solution available. Please run solve() first.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.solution,
            triangles=self.mesh.elements,
            cmap="viridis",
            edgecolor="none",
        )
        ax.set_title("FEM 3D Solution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("u")
        plt.show()

    def plotError2d(self, exact_solution):
        if self.solution is None:
            raise ValueError("No solution available. Please run solve() first.")

        # Compute error at each node
        exact_values = np.array([exact_solution(x, y) for x, y in self.mesh.points])
        error = np.abs(self.solution - exact_values)

        plt.tricontourf(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.elements,
            error,
            levels=14,
            cmap="inferno",
        )
        plt.triplot(
            self.mesh.points[:, 0],
            self.mesh.points[:, 1],
            self.mesh.elements,
            color="black",
            alpha=0.3,
            linewidth=0.5,
        )
        plt.colorbar(label="Absolute Error |u - u_exact|")
        plt.title("FEM Solution Error")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()

    def compute_max_error(self, exact_solution):
        if self.solution is None:
            raise ValueError("No solution available. Please run solve() first.")

        exact_values = np.array([exact_solution(x, y) for x, y in self.mesh.points])
        error = np.abs(self.solution - exact_values)
        max_error = np.max(error)
        return max_error
