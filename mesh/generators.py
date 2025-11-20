import numpy as np
from scipy.io import loadmat


def generatePoints2dUnitSquare(h):
    if h <= 0 or h > 1:
        raise ValueError("h must be in the interval (0, 1].")

    if 1/h != int(1/h):
        print("WARNING: 1/h not INT, rounding 1/h to nearest value")
    N = int(1 / h)+1
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    
    return points


def generatePoints2dRectangle(xmin, xmax, ymin, ymax, h):
    if h <= 0 or h > min(xmax - xmin, ymax - ymin):
        raise ValueError("h must be in the interval (0, min(width, height)].")

    x = np.arange(xmin, xmax + h, h)
    y = np.arange(ymin, ymax + h, h)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    return points


def getPointsFromMatFile(filename, mesh_index=0):
    # Load points from a .mat file 
    # first column is coordinate matrix, second column is element connectivity (triangles) and last column is connectivity (free edges)
    mat_data = loadmat(filename)
    data = mat_data['mesh']
    print(f"Available meshes in file: {data.shape[0]}")
    print(f"Using mesh index: {mesh_index}")
    coords = (data[mesh_index][0])
    elements = (data[mesh_index][1]) -1
    edges = (data[mesh_index][2]) -1

    return coords, elements, edges



def generatePoints2dCircle(radius, h):
    if h <= 0 or h > radius:
        raise ValueError("h must be in the interval (0, radius].")
    
    points = [[0, 0]]  # start with center point
    
    # concentric rings
    r_values = np.arange(h, radius + h, h)
    for r in r_values:
        circumference = 2 * np.pi * r
        n_points = max(int(np.ceil(circumference / h)), 1)
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.extend(np.vstack([x, y]).T)
    
    points = np.array(points)
    
    
    return points