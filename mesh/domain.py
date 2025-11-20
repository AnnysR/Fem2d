import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


class Mesh2D:
    def __init__(self, points = None):
        if points is None:
            return
        self.points = self.removeNearDuplicates(np.asarray(points))
        self.tri = Delaunay(self.points)
        self.elements = self.tri.simplices
        self.edges = self._computeEdges()
        self.boundary_nodes = self.findBoundaryNodes()


    def init_from_mat(self, coords, elements, free_edges):
        self.points = self.removeNearDuplicates(np.asarray(coords))
        self.elements = np.asarray(elements) 
        self.edges = np.asarray(free_edges)
        self.boundary_nodes = self.findBoundaryNodes()
        
        print("Mesh initialized from .mat file:")
        print(f"  Number of points: {self.points.shape[0]}")
        print(f"  Number of elements: {self.elements.shape[0]}")
        print(f"  Number of edges: {self.edges.shape[0]}")
        
        
    def removeNearDuplicates(self, points, tol=1e-12):
        unique_points = []
        for p in points:
            if not unique_points:
                unique_points.append(p)
            else:
                dists = np.linalg.norm(np.array(unique_points) - p, axis=1)
                if np.all(dists > tol):
                    unique_points.append(p)
        return np.array(unique_points)

    def _computeEdges(self):
        edges = np.vstack(
            [
                self.elements[:, [0, 1]],
                self.elements[:, [1, 2]],
                self.elements[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        unique_edges = np.unique(edges, axis=0)
        return unique_edges

    def findBoundaryNodes(self):
        edges = np.vstack(
            [
                self.elements[:, [0, 1]],
                self.elements[:, [1, 2]],
                self.elements[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = edges[counts == 1]

        boundary_nodes = np.unique(boundary_edges)
        return boundary_nodes

    def plot(self):
        nbTriangles = self.elements.shape[0]
        plt.triplot(
            self.points[:, 0],
            self.points[:, 1],
            self.elements,
            color="gray",
            alpha=1,
            linewidth=0.5,
        )

        edges_all = np.vstack(
            [
                self.elements[:, [0, 1]],
                self.elements[:, [1, 2]],
                self.elements[:, [2, 0]],
            ]
        )
        edges_sorted = np.sort(edges_all, axis=1)
        uniq_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        boundary_edges = uniq_edges[counts == 1]

        for edge in boundary_edges:
            x = self.points[edge, 0]
            y = self.points[edge, 1]
            plt.plot(x, y, color="black", linewidth=2.5, zorder=3)

        for i in range(nbTriangles):
            x = self.points[self.elements[i, :], 0]
            y = self.points[self.elements[i, :], 1]
            xc = np.mean(x)
            yc = np.mean(y)
            if nbTriangles < 50:
                plt.text(xc, yc, str(i), color="blue", fontsize=12)
        if nbTriangles < 50:
            for i in range(self.points.shape[0]):
                plt.text(
                    self.points[i, 0],
                    self.points[i, 1],
                    str(i),
                    color="green",
                    fontsize=12,
                )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.title("2D Mesh")
        plt.show()
