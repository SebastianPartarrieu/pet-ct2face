import numpy as np
from skimage.filters import threshold_otsu
import cc3d
import open3d as o3d
from skimage.color import rgb2gray

def find_skin(data, is_pet=False):
    if is_pet:
        otsu = np.percentile(data, 95)
        data_bin = (data > otsu)
    else:
        otsu = threshold_otsu(data)
        data_bin = (data >= otsu)
        labels = cc3d.connected_components(data_bin)
    return otsu, labels

def generate_morpho(lab):
    morpho = lab.copy()
    unique_lab, counts = np.unique(lab, return_counts=True)
    skin = unique_lab[np.argsort(counts)[-2]] # first largest comp is background
    morpho = (morpho == skin).astype(float)
    return morpho

def create_mesh(verts, faces, vox_dim, height, n_it=5):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh = mesh.filter_smooth_laplacian(n_it)
    mesh.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[200*vox_dim[0], 0, height*vox_dim[-1]],
        eye=[200*vox_dim[0], 400*vox_dim[1], height*1.01*vox_dim[-1]],
        up=[0, 1, 0],
        width_px=200,
        height_px=200,
    )
    ans = scene.cast_rays(rays)
    proj = np.abs(ans['primitive_normals'].numpy())
    proj_g = (rgb2gray(proj)*255).astype(np.uint8)
    return proj_g