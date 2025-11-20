import torch
from tsr.system import TSR
from PIL import Image
import os

from fine_tuning import inject_lora_into_model

def load_image(path):
    return Image.open(path).convert("RGB")

def apply_lora_weights(model, lora_path):
    print(f"[INFO] Cargando pesos LoRA desde: {lora_path}")
    sd = torch.load(lora_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[LoRA] keys cargadas:", len(sd))
    print("[LoRA] Missing:", missing)
    print("[LoRA] Unexpected:", unexpected)

def run_inference(image_path, lora_path, repo_root="."):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Modelo base
    model = TSR.from_pretrained(repo_root, config_name="config.yaml", weight_name="model.ckpt")
    model.to(device)

    # 2. Inyectar LoRA *exactamente igual que en el entrenamiento*
    inject_lora_into_model(model.backbone, r=16, alpha=16, dropout=0.05)

    # 3. Cargar pesos LoRA
    apply_lora_weights(model, lora_path)

    # 4. Cargar imagen
    img = load_image(image_path)

    # 5. Obtener scene_code
    scene = model(img, device=device)
    if isinstance(scene, dict):
        scene_code = scene.get("scene_code") or scene.get("latent")
    else:
        scene_code = scene
    scene_code = scene_code.to(device)

    # 6. Reconstrucción 3D
    mesh = model.extract_mesh(scene_code, has_vertex_color=False)

    if isinstance(mesh, list):
        for i, m in enumerate(mesh):
            out_path = f"mesh_lora_{i}.obj"
            m.export(out_path)
            print(f"[OK] Mesh exportado: {out_path}")
    else:
        mesh.export("mesh_lora.obj")
        print("[OK] Mesh exportado: mesh_lora.obj")
    # 7. Guardar
    out_path = "mesh_lora.obj"
    print(f"[OK] Mesh exportado a {out_path}")

if __name__ == "__main__":
    run_inference(
        image_path="examples/chair.png",
        lora_path="outputs_lora/lora_weights.pth",
        repo_root="."
    )

