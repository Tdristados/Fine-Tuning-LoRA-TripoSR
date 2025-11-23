#!/usr/bin/env python
"""
Script para generar mallas 3D (.obj, .ply, etc.) con TripoSR, usando:

- El modelo base  (solo model.ckpt)
- El modelo con LoRA (model.ckpt + weights_lora/lora_weights.pth)

Entrada:
  --images_dir  -> carpeta con imágenes (o una sola imagen)
Salida:
  --output_dir  -> carpeta donde se guardarán las mallas <nombre>.obj

Ejemplos de uso
---------------

# 1) Modelo base (original)
python generate_mesh_triposr.py \
  --images_dir ~/data/ComputerVision/data/dataTest/images \
  --output_dir outputs/base_meshes \
  --pretrained_name_or_path . \
  --config_name config.yaml \
  --weight_name model.ckpt \
  --model_type base

# 2) Modelo fine-tuneado con LoRA
python generate_mesh_triposr.py \
  --images_dir ~/data/ComputerVision/data/dataTest/images \
  --output_dir outputs/lora_meshes \
  --pretrained_name_or_path . \
  --config_name config.yaml \
  --weight_name model.ckpt \
  --model_type lora \
  --lora_weights weights_lora/lora_weights.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


# ---------------------------------------------------------
# 1. Argumentos
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generar mallas 3D con TripoSR (modelo base o modelo LoRA)."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Ruta a una carpeta de imágenes o a una sola imagen.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Carpeta donde se guardarán las mallas (.obj, .ply, etc.).",
    )
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        default="stabilityai/TripoSR",
        help=(
            "Nombre del modelo en HF o carpeta local con config.yaml y model.ckpt. "
            "Si estás en el repo local, típicamente es '.'."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="config.yaml",
        help="Nombre o ruta del archivo de configuración de TripoSR.",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default="model.ckpt",
        help="Nombre o ruta del checkpoint base (model.ckpt).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "lora"],
        default="base",
        help="Escoge 'base' para el modelo original o 'lora' para el modelo fine-tuneado.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Ruta a weights_lora/lora_weights.pth (solo si --model_type=lora).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo: 'auto', 'cpu', 'cuda' o 'cuda:0', etc.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Chunk size del renderer NeRF (reduce picos de memoria).",
    )
    parser.add_argument(
        "--mesh_format",
        type=str,
        default="obj",
        choices=["obj", "ply", "stl", "glb"],
        help="Formato de salida de la malla.",
    )
    parser.add_argument(
        "--no_remove_bg",
        action="store_true",
        help="Si se pasa, NO se elimina el fondo de la imagen.",
    )
    parser.add_argument(
        "--foreground_ratio",
        type=float,
        default=0.85,
        help="Escala del objeto tras recorte de fondo.",
    )
    parser.add_argument(
        "--has_vertex_color",
        action="store_true",
        help="Si se pasa, exporta malla con color de vértices (más pesado).",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 2. Utilidades
# ---------------------------------------------------------

def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_triposr_model(
    pretrained_name_or_path,
    config_name,
    weight_name,
    device,
    chunk_size,
    model_type,
    lora_weights=None,
):
    """
    Carga el modelo base de TripoSR y, si se pide, aplica pesos LoRA.
    """
    print(f"[INFO] Cargando modelo TripoSR desde '{pretrained_name_or_path}'...")
    model = TSR.from_pretrained(
        pretrained_name_or_path,
        config_name=config_name,
        weight_name=weight_name,
    )
    model.renderer.set_chunk_size(chunk_size)
    model.to(device)

    if model_type == "lora":
        if lora_weights is None:
            raise ValueError(
                "Seleccionaste --model_type=lora pero no pasaste --lora_weights."
            )
        print(f"[LoRA] Cargando pesos LoRA desde: {lora_weights}")
        state = torch.load(lora_weights, map_location=device)

        # Intentamos ser flexibles con el formato guardado
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[LoRA] load_state_dict OK "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if len(unexpected) > 0:
            print("[LoRA] Aviso: hay claves inesperadas (ej. optimizer_state, etc.). "
                  "Si no hay errores de dimensiones, puedes ignorarlo.")

    model.eval()
    return model


def preprocess_image(
    image_path,
    remove_bg_flag,
    foreground_ratio,
    rembg_session=None,
):
    image = Image.open(image_path).convert("RGBA")

    if remove_bg_flag:
        if rembg_session is None:
            rembg_session = rembg.new_session()
        image = remove_background(image, rembg_session)

    image = resize_foreground(image, foreground_ratio)

    if image.mode == "RGBA":
        img_np = np.array(image).astype(np.float32) / 255.0
        rgb = img_np[:, :, :3]
        alpha = img_np[:, :, 3:4]
        img_np = rgb * alpha + (1.0 - alpha) * 0.5  # fondo gris
        image = Image.fromarray((img_np * 255.0).astype(np.uint8))
    else:
        image = image.convert("RGB")

    return image


def collect_image_paths(images_dir: Path):
    if images_dir.is_file():
        return [images_dir]
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        p for p in images_dir.glob("**/*") if p.suffix.lower() in exts
    )


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    args = parse_args()
    device = resolve_device(args.device)

    images_dir = Path(args.images_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes en '{images_dir}'. "
            "Extensiones soportadas: .png, .jpg, .jpeg, .webp"
        )

    print(f"[INFO] Encontradas {len(image_paths)} imágenes.")
    print(f"[INFO] Guardando mallas en: {output_root.resolve()}")

    model = load_triposr_model(
        pretrained_name_or_path=args.pretrained_name_or_path,
        config_name=args.config_name,
        weight_name=args.weight_name,
        device=device,
        chunk_size=args.chunk_size,
        model_type=args.model_type,
        lora_weights=args.lora_weights,
    )

    rembg_session = None if args.no_remove_bg else rembg.new_session()

    for idx, img_path in enumerate(image_paths):
        img_id = img_path.stem  # nombre sin extensión
        print(f"\n[INFO] Procesando imagen {idx + 1}/{len(image_paths)}: {img_path}")

        image = preprocess_image(
            img_path,
            remove_bg_flag=not args.no_remove_bg,
            foreground_ratio=args.foreground_ratio,
            rembg_session=rembg_session,
        )

        # Guardar la imagen preprocesada (opcional, por trazabilidad)
        pre_img_path = output_root / f"{img_id}_input.png"
        image.save(pre_img_path)

        with torch.no_grad():
            scene_codes = model([image], device=device)
            meshes = model.extract_mesh(
                scene_codes, has_vertex_color=args.has_vertex_color
            )

        # OJO: aquí guardamos directamente <nombre>.obj en output_root
        mesh_path = output_root / f"{img_id}.{args.mesh_format}"
        meshes[0].export(mesh_path.as_posix())
        print(f"[OK] Malla exportada: {mesh_path}")

    print("\n[FIN] Todas las mallas han sido generadas.")


if __name__ == "__main__":
    main()
