import trimesh
import numpy as np

# Test mesh with 1 cube
r = 0.5
cube_verts = np.array([
    [-r, -r, -r],
    [ r, -r, -r],
    [ r,  r, -r],
    [-r,  r, -r],
    [-r, -r,  r],
    [ r, -r,  r],
    [ r,  r,  r],
    [-r,  r,  r]
], dtype=np.float32)

cube_faces = np.array([
    [0, 1, 2], [0, 2, 3], # front
    [1, 5, 6], [1, 6, 2], # right
    [5, 4, 7], [5, 7, 6], # back
    [4, 0, 3], [4, 3, 7], # left
    [4, 5, 1], [4, 1, 0], # bottom
    [3, 2, 6], [3, 6, 7]  # top
], dtype=np.int32)

colors = np.array([[255, 0, 0, 255]] * 8, dtype=np.uint8)

mesh = trimesh.Trimesh(vertices=cube_verts, faces=cube_faces, vertex_colors=colors)

try:
    mesh.export('test.glb')
    print("GLB export OK")
except Exception as e:
    print(f"GLB export failed: {e}")

try:
    mesh.export('test.usdz')
    print("USDZ export OK")
except Exception as e:
    print(f"USDZ export failed: {e}")

try:
    mesh.export('test.obj')
    print("OBJ export OK")
except Exception as e:
    print(f"OBJ export failed: {e}")

