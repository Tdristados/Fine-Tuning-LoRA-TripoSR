# data_processing.py
# -*- coding: utf-8 -*-
"""
Render multivista con trimesh + pyrender
Genera por objeto:
  - IMG/000.png..005.png
  - MASK/000.png..005.png
  - Depth/000.npy..005.npy
  - cams.json

Requisitos:
  pip install trimesh pyrender numpy pillow tqdm

Uso:
  python data_processing.py --objs_dir "./obj" --output_root "./dataFT" --img_size 512 --yfov 50
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import trimesh
import pyrender

# --- (Opcional) si tienes problemas de backend en servidor/headless, descomenta UNA:
# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"     # GPU headless (NVIDIA)
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # CPU (OSMesa)

# -----------------------------
# Utils
# -----------------------------
def center_and_unit_scale(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centra el modelo en el origen y escala para que su mayor dimensión sea 1."""
    mesh = mesh.copy()
    if mesh.is_empty:
        return mesh
    mesh.apply_translation(-mesh.centroid)
    extent = float(np.max(mesh.extents)) if np.max(mesh.extents) > 0 else 1.0
    mesh.apply_scale(1.0 / extent)
    _ = mesh.vertex_normals  # fuerza cálculo de normales si falta
    return mesh

def yfov_to_intrinsics(w: int, h: int, yfov_deg: float) -> np.ndarray:
    """Intrínseca K 3x3 con píxeles cuadrados y centro óptico en w/2,h/2."""
    f = (h / 2.0) / math.tan(math.radians(yfov_deg) / 2.0)
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=np.float64)
    return K

def view_poses(distance: float) -> Dict[str, np.ndarray]:
    """
    Matrices camera->world (C2W) 4x4 para: front, right, back, left, top, bottom.
    Convención OpenGL (pyrender): la cámara mira a -Z en coords de cámara.
    Estas C2W colocan la cámara mirando al origen desde una esfera de radio d.
    """
    d = float(distance)
    return {
        "front":  np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, d],
                            [0, 0, 0, 1]], dtype=np.float64),
        "right":  np.array([[ 0, 0, 1, d],
                            [ 0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [ 0, 0, 0, 1]], dtype=np.float64),
        "back":   np.array([[1, 0, 0, 0],
                            [0,-1, 0, 0],
                            [0, 0,-1,-d],
                            [0, 0, 0, 1]], dtype=np.float64),
        "left":   np.array([[0, 0,-1,-d],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64),
        "top":    np.array([[1, 0, 0, 0],
                            [0, 0,-1,-d],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.float64),
        "bottom": np.array([[ 1, 0, 0, 0],
                            [ 0, 0, 1, d],
                            [ 0,-1, 0, 0],
                            [ 0, 0, 0, 1]], dtype=np.float64),
    }

def c2w_to_w2c(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte camera->world (C2W) a world->camera (R,T)."""
    Minv = np.linalg.inv(M)
    R = Minv[:3, :3]
    T = Minv[:3, 3:4]
    return R, T

def make_mask(depth: np.ndarray) -> np.ndarray:
    """Máscara binaria desde depth de pyrender: fondo suele ser 0."""
    mask = (depth > 0) & np.isfinite(depth)
    return (mask.astype(np.uint8) * 255)

def depth_stats(depth: np.ndarray) -> str:
    nz = int((depth > 0).sum())
    if nz == 0:
        return "valid_px=0"
    return f"valid_px={nz}, min={float(depth[depth>0].min()):.5f}, max={float(depth.max()):.5f}"

# -----------------------------
# Core
# -----------------------------
def render_object(obj_path: Path,
                  out_root: Path,
                  img_size: int = 512,
                  yfov_deg: float = 50.0) -> None:
    name = obj_path.stem
    out_dir = out_root / name
    (out_dir / "IMG").mkdir(parents=True, exist_ok=True)
    (out_dir / "MASK").mkdir(parents=True, exist_ok=True)
    (out_dir / "Depth").mkdir(parents=True, exist_ok=True)

    # 1) Cargar malla SIN procesar para NO perder materiales/UV
    loaded = trimesh.load_mesh(str(obj_path), process=False)

    # Si viene como Scene, convertir a Trimesh preservando visuales cuando sea posible
    if isinstance(loaded, trimesh.Scene):
        try:
            tri_list = loaded.to_trimesh()
            mesh = trimesh.util.concatenate(tri_list) if isinstance(tri_list, (list, tuple)) else tri_list
        except Exception:
            mesh = trimesh.util.concatenate(tuple(loaded.dump().geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"No se pudo convertir a Trimesh: {obj_path}")

    mesh = center_and_unit_scale(mesh)

    # 2) Distancia de cámara para encuadre con margen
    verts = np.asarray(mesh.vertices)
    r = float(np.linalg.norm(verts, axis=1).max()) if len(verts) else 0.5
    phi = math.radians(yfov_deg) / 2.0
    d = max(2.5, (r / max(math.sin(phi), 1e-6)) * 1.15)

    # 3) Escena pyrender (NO forzar material para respetar texturas)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Cámara / renderer / luz
    camera = pyrender.PerspectiveCamera(
        yfov=math.radians(yfov_deg), aspectRatio=1.0, znear=0.05, zfar=50.0
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=img_size, viewport_height=img_size)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    # 4) Vistas e intrínseca
    K = yfov_to_intrinsics(img_size, img_size, yfov_deg)
    poses = view_poses(distance=d)
    view_order = ["front", "right", "back", "left", "top", "bottom"]

    cams = []
    for idx, view in enumerate(view_order):
        c2w = poses[view]
        cam_node = scene.add(camera, pose=c2w)
        scene.main_camera_node = cam_node     # <- CLAVE para tu versión de pyrender
        light_node = scene.add(light, pose=c2w)

        # Render SIN flags y SIN camera_node (tu versión no lo soporta)
        color, depth = renderer.render(scene)

        # Debug ligero en consola
        #print(f"[{name} {view}] {depth_stats(depth)}")

        # Guardar imagen RGB
        Image.fromarray(color[:, :, :3]).save(out_dir / "IMG" / f"{idx:03d}.png")

        # Guardar depth (float32)
        np.save(out_dir / "Depth" / f"{idx:03d}.npy", depth.astype(np.float32))

        # Guardar máscara desde depth
        Image.fromarray(make_mask(depth), mode="L").save(out_dir / "MASK" / f"{idx:03d}.png")

        # Extrínseca world->camera
        R, T = c2w_to_w2c(c2w)
        cams.append({
            "file_id": f"{idx:03d}",
            "view": view,
            "width": img_size,
            "height": img_size,
            "yfov_deg": float(yfov_deg),
            "K": K.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "depth_encoding": "pyrender_depth>0_is_fg"
        })

        # Limpiar nodos de esta vista
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

    # 5) Guardar cámaras
    with open(out_dir / "cams.json", "w", encoding="utf-8") as f:
        json.dump(cams, f, indent=2)

def process_folder(objs_dir: Path,
                   output_root: Path,
                   img_size: int = 512,
                   yfov_deg: float = 50.0) -> None:
    obj_paths = sorted(objs_dir.rglob("*.obj"))
    if not obj_paths:
        raise SystemExit(f"No se encontraron .obj en: {objs_dir}")
    for p in tqdm(obj_paths, desc="Renderizando", unit="obj"):
        try:
            render_object(p, output_root, img_size=img_size, yfov_deg=yfov_deg)
        except Exception as e:
            tqdm.write(f"ERROR {p.stem}: {e}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs_dir", required=True, type=Path, help="Carpeta con .obj (recursiva).")
    ap.add_argument("--output_root", default=Path("./dataFT"), type=Path, help="Raíz de salida.")
    ap.add_argument("--img_size", default=512, type=int, help="Resolución (cuadrada).")
    ap.add_argument("--yfov", default=50.0, type=float, help="FOV vertical (grados).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    process_folder(args.objs_dir, args.output_root, img_size=args.img_size, yfov_deg=args.yfov)
    print(f"\n✅ Dataset generado en: {args.output_root}")
