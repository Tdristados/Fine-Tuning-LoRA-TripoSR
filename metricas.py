#!/usr/bin/env python
"""
Evaluación de mallas TripoSR (base vs LoRA) usando Chamfer Distance y F-Score.

Uso típico
----------

# Solo modelo base
python metricas_triposr.py \
  --gt_dir ~/data/ComputerVision/data/dataTest/meshes_gt \
  --base_dir outputs/base_meshes

# Base + LoRA
python metricas_triposr.py \
  --gt_dir ~/data/ComputerVision/data/dataTest/meshes_gt \
  --base_dir outputs/base_meshes \
  --lora_dir outputs/lora_meshes

Parámetros opcionales:
  --n_points 20000           # número de puntos muestreados por malla
  --thresholds 0.1 0.2 0.5   # umbrales para F-Score
  --ext obj                  # extensión de las mallas (.obj, .ply, etc.)
"""

import os
import glob
import argparse
import numpy as np
import torch
import trimesh


# ---------------------------------------------------------
# Utilidades para carga y muestreo de mallas
# ---------------------------------------------------------

def load_and_sample_mesh(mesh_path, n_points=10000):
    """
    Carga una malla (.obj, .ply, etc.) y devuelve n_points puntos de su superficie
    como tensor (N, 3) en float32.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    # A veces se carga como Scene; concatenamos a un solo Trimesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    points = torch.from_numpy(points).float()  # (N, 3)
    return points


def normalize_points(points):
    """
    Normaliza el point cloud a un rango aproximadamente [-0.5, 0.5]^3
    para que los thresholds 0.1, 0.2, 0.5 sean comparables entre objetos.
    """
    center = points.mean(dim=0, keepdim=True)
    points = points - center
    max_range = points.abs().max().clamp(min=1e-6)
    points = points / (2 * max_range)
    return points


# ---------------------------------------------------------
# Chamfer Distance y F-Score
# ---------------------------------------------------------

def chamfer_distance(pred_pts, gt_pts):
    """
    Chamfer-L1 bidireccional entre dos nubes de puntos.

    pred_pts: (N, 3)
    gt_pts:   (M, 3)
    """
    pred = pred_pts.unsqueeze(0)  # (1, N, 3)
    gt = gt_pts.unsqueeze(0)      # (1, M, 3)

    # Distancias Euclidianas entre todos los pares
    dists = torch.cdist(pred, gt, p=2)  # (1, N, M)

    # Distancia mínima de cada punto predicho a la GT
    min_pred_to_gt, _ = dists.min(dim=2)  # (1, N)
    # Distancia mínima de cada punto GT a la predicción
    min_gt_to_pred, _ = dists.min(dim=1)  # (1, M)

    cd = min_pred_to_gt.mean() + min_gt_to_pred.mean()
    return cd.item()


def fscore(pred_pts, gt_pts, tau=0.1):
    """
    F-score@tau estándar:

      P = fracción de puntos predichos a distancia < tau de la GT
      R = fracción de puntos GT     a distancia < tau de la predicción
      F = 2 * P * R / (P + R)
    """
    pred = pred_pts.unsqueeze(0)  # (1, N, 3)
    gt = gt_pts.unsqueeze(0)      # (1, M, 3)

    dists = torch.cdist(pred, gt, p=2)  # (1, N, M)
    min_pred_to_gt, _ = dists.min(dim=2)  # (1, N)
    min_gt_to_pred, _ = dists.min(dim=1)  # (1, M)

    precision = (min_pred_to_gt < tau).float().mean().item()
    recall    = (min_gt_to_pred < tau).float().mean().item()

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------
# Evaluación sobre un directorio de mallas
# ---------------------------------------------------------

def evaluate_model(gt_mesh_dir, pred_mesh_dir, n_points=10000,
                   thresholds=(0.1, 0.2, 0.5), ext=".obj"):
    """
    Evalúa un modelo comparando sus mallas predichas con las GT.

    gt_mesh_dir:    carpeta con mallas GT
    pred_mesh_dir:  carpeta con mallas predichas
    ext:            extensión de las mallas ('.obj', '.ply', etc.)
    """
    gt_mesh_dir = os.path.expanduser(gt_mesh_dir)
    pred_mesh_dir = os.path.expanduser(pred_mesh_dir)

    pattern = os.path.join(gt_mesh_dir, "*" + ext)
    gt_paths = sorted(glob.glob(pattern))
    if not gt_paths:
        raise FileNotFoundError(f"No se encontraron {ext} en {gt_mesh_dir}")

    mesh_ids = [os.path.splitext(os.path.basename(p))[0] for p in gt_paths]

    all_cd = []
    all_fs = {tau: [] for tau in thresholds}

    print(f"[INFO] Evaluando {len(mesh_ids)} mallas:")
    print(f"       GT   : {gt_mesh_dir}")
    print(f"       Pred : {pred_mesh_dir}")
    print(f"       Ext  : {ext}\n")

    for mid in mesh_ids:
        gt_path = os.path.join(gt_mesh_dir,   mid + ext)
        pred_path = os.path.join(pred_mesh_dir, mid + ext)

        if not os.path.exists(pred_path):
            print(f"[WARN] No se encontró predicción para '{mid}' en {pred_mesh_dir}, se omite.")
            continue

        gt_pts = load_and_sample_mesh(gt_path, n_points)
        pred_pts = load_and_sample_mesh(pred_path, n_points)

        gt_pts = normalize_points(gt_pts)
        pred_pts = normalize_points(pred_pts)

        cd_val = chamfer_distance(pred_pts, gt_pts)
        all_cd.append(cd_val)

        for tau in thresholds:
            f_val = fscore(pred_pts, gt_pts, tau=tau)
            all_fs[tau].append(f_val)

    if not all_cd:
        raise RuntimeError("No se pudo evaluar ninguna malla (faltan predicciones).")

    mean_cd = float(np.mean(all_cd))
    mean_fs = {tau: float(np.mean(vals)) for tau, vals in all_fs.items()}

    return mean_cd, mean_fs


# ---------------------------------------------------------
# Argumentos y main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluar mallas TripoSR (Chamfer Distance + F-Score)."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Ruta carpeta GT (meshes_gt con .obj).",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Ruta carpeta predicciones BASE (outputs/base_meshes).",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Ruta carpeta predicciones LORA (outputs/lora_meshes). Opcional.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10000,
        help="Número de puntos muestreados por malla (default: 10000).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5],
        help="Umbrales para F-Score (default: 0.1 0.2 0.5).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="obj",
        help="Extensión de las mallas sin punto (default: obj).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = tuple(args.thresholds)
    ext = "." + args.ext.lstrip(".")

    print("=== Evaluación TripoSR (Chamfer Distance + F-Score) ===\n")

    # ----------------- MODELO BASE -----------------
    print("--------------------------------------")
    print("Evaluando MODELO BASE...")
    print("--------------------------------------")
    cd_base, fs_base = evaluate_model(
        args.gt_dir,
        args.base_dir,
        n_points=args.n_points,
        thresholds=thresholds,
        ext=ext,
    )
    print("\n=== Resultados TripoSR BASE ===")
    print(f"Chamfer Distance (CD): {cd_base:.6f}")
    for tau in thresholds:
        print(f"F-Score@{tau:.2f}: {fs_base[tau]:.4f}")

    cd_lora = None
    fs_lora = None

    # ----------------- MODELO LORA (opcional) -----------------
    if args.lora_dir is not None and args.lora_dir != "":
        print("\n--------------------------------------")
        print("Evaluando MODELO LORA...")
        print("--------------------------------------")
        cd_lora, fs_lora = evaluate_model(
            args.gt_dir,
            args.lora_dir,
            n_points=args.n_points,
            thresholds=thresholds,
            ext=ext,
        )
        print("\n=== Resultados TripoSR + LoRA ===")
        print(f"Chamfer Distance (CD): {cd_lora:.6f}")
        for tau in thresholds:
            print(f"F-Score@{tau:.2f}: {fs_lora[tau]:.4f}")

    # ----------------- COMPARACIÓN -----------------
    if cd_lora is not None:
        print("\n--------------------------------------")
        print("Comparación LoRA vs BASE (Δ = LoRA - BASE)")
        print("--------------------------------------")
        print(f"ΔCD = {cd_lora - cd_base:.6f}  (negativo = LoRA mejora)")
        for tau in thresholds:
            delta_fs = fs_lora[tau] - fs_base[tau]
            print(f"ΔF-Score@{tau:.2f} = {delta_fs:+.4f}  (positivo = LoRA mejora)")

    print("\n[FIN] Evaluación completada.")


if __name__ == "__main__":
    main()
