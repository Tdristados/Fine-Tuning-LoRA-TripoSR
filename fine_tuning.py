
# fine_tuning.py (parche BHWC + pérdidas robustas CHW)
import os, json, math, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.enabled = False 
from tsr.system import TSR
from tsr.utils import get_ray_directions, get_rays

# ---------------- LoRA ----------------
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.A = nn.Linear(self.base.in_features, r, bias=False)
        self.B = nn.Linear(r, self.base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.dropout(self.B(self.A(x))) * self.scaling

def _should_wrap_lora(module_path: str) -> bool:
    pats = (".to_q", ".to_k", ".to_v", ".to_out.0", "q_proj", "k_proj", "v_proj", "out_proj", ".qkv")
    return any(p in module_path for p in pats)

def inject_lora_into_model(model: nn.Module, where: Optional[nn.Module]=None, r=16, alpha=16, dropout=0.0) -> List[str]:
    if where is None:
        where = model
    rep = []
    for full_name, module in list(where.named_modules()):
        if isinstance(module, nn.Linear) and _should_wrap_lora(full_name):
            if full_name == "":
                continue
            parts = full_name.split(".")
            parent = where
            for k in parts[:-1]:
                parent = parent._modules[k]
            key = parts[-1]
            parent._modules[key] = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            rep.append(full_name)
    return rep

def lora_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for n,p in model.named_parameters() if ("A.weight" in n or "B.weight" in n) and p.requires_grad]

def save_lora(model: nn.Module, path: str):
    sd = {n: p for n,p in model.state_dict().items() if ("A.weight" in n or "B.weight" in n)}
    torch.save(sd, path)

# --------------- Dataset / Cámara ---------------
def _pil_open_rgb(path: str, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)

def _load_mask(path: str, size: int) -> torch.Tensor:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return torch.from_numpy((np.array(m) > 127).astype(np.float32))[None, ...]  # (1,H,W)

def _to_tensor_uint(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1)  # (3,H,W)

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Normaliza tensores de rayos a (B,H,W,3)."""
    if x.dim() == 4:
        if x.shape[-1] == 3:     # (B,H,W,3)
            return x
        if x.shape[1] == 3:      # (B,3,H,W) -> (B,H,W,3)
            return x.permute(0,2,3,1).contiguous()
        idx3 = [i for i,s in enumerate(x.shape) if s == 3]
        if len(idx3) == 1:
            order = [0] + [i for i in range(1,4) if i != idx3[0]] + [idx3[0]]
            return x.permute(*order).contiguous()
        raise RuntimeError(f"Forma de rayos 4D no reconocida: {tuple(x.shape)}")
    if x.dim() == 3:
        if x.shape[-1] == 3:     # (H,W,3) -> (1,H,W,3)
            return x.unsqueeze(0)
        if x.shape[0] == 3:      # (3,H,W) -> (1,H,W,3)
            return x.permute(1,2,0).unsqueeze(0).contiguous()
        raise RuntimeError(f"Forma de rayos 3D no reconocida: {tuple(x.shape)}")
    raise RuntimeError(f"Dimensión de rayos no soportada: {x.dim()}")

def _to_chw3(x: torch.Tensor) -> torch.Tensor:
    """Convierte cualquier layout RGB a (3,H,W) y elimina batch si existe."""
    if x.dim() == 4:
        if x.shape[-1] == 3:      # (B,H,W,3) -> (B,3,H,W)
            x = x.permute(0,3,1,2).contiguous()
        # ahora puede ser (B,3,H,W)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x
    if x.dim() == 3 and x.shape[-1] == 3:  # (H,W,3) -> (3,H,W)
        return x.permute(2,0,1).contiguous()
    return x  # ya (3,H,W)

def _to_chw1(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convierte máscara a (1,H,W). Acepta (H,W), (1,H,W), (H,W,1), (1,H,W,1), (B,1,H,W), (B,H,W,1)."""
    if m is None:
        return None
    if m.dim() == 2:            # (H,W)
        return m.unsqueeze(0)
    if m.dim() == 3:
        if m.shape[0] == 1:     # (1,H,W)
            return m
        if m.shape[-1] == 1:    # (H,W,1) -> (1,H,W)
            return m.permute(2,0,1).contiguous()
    if m.dim() == 4:
        if m.shape[0] == 1 and m.shape[1] == 1:   # (1,1,H,W)
            return m.squeeze(0)
        if m.shape[0] == 1 and m.shape[-1] == 1:  # (1,H,W,1) -> (1,1,H,W) -> (1,H,W)
            return m.permute(0,3,1,2).squeeze(0).contiguous()
    raise RuntimeError(f"Máscara con forma no soportada: {tuple(m.shape)}")

def _cam_from_json(cam: Dict[str, Any], H_out: int, W_out: int, auto_flip: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    # Intrinsics
    if "K" in cam:
        K = torch.tensor(cam["K"], dtype=torch.float32)
        fx = float(K[0,0]); fy = float(K[1,1]); cx = float(K[0,2]); cy = float(K[1,2])
        W_src = int(cam.get("width", W_out)); H_src = int(cam.get("height", H_out))
        sx = W_out / float(W_src); sy = H_out / float(H_src)
        fx *= sx; fy *= sy
        cx = (cx + 0.5) * sx - 0.5
        cy = (cy + 0.5) * sy - 0.5
    else:
        yfov = float(cam.get("yfov_deg", cam.get("yfov", 50.0)))
        fy = H_out / (2.0 * math.tan(math.radians(yfov) / 2.0))
        fx = fy * float(cam.get("aspect", 1.0))
        cx, cy = W_out/2.0, H_out/2.0

    # Extrinsics (R,T como WORLD->CAM: invertimos a c2w)
    if "c2w" in cam:
        c2w = torch.tensor(cam["c2w"], dtype=torch.float32)
        if c2w.numel() == 12:
            c2w = torch.cat([c2w.view(3,4), torch.tensor([[0,0,0,1]], dtype=torch.float32)], dim=0)
        elif c2w.numel() == 16:
            c2w = c2w.view(4,4)
    elif "R" in cam and "T" in cam:
        R = torch.tensor(cam["R"], dtype=torch.float32).view(3,3)
        T = torch.tensor(cam["T"], dtype=torch.float32).reshape(3)
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3,:3] = R; w2c[:3, 3] = T
        c2w = torch.inverse(w2c)
    else:
        raise ValueError("cams.json: no encuentro extrínsecas (usa 'c2w' o 'R'+'T').")

    # Rays
    directions = get_ray_directions(H_out, W_out, (fx, fy), (cx, cy), use_pixel_centers=True, normalize=True)
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)
    rays_o, rays_d = _to_bhwc(rays_o), _to_bhwc(rays_d)

    # Auto-flip: asegurar que el rayo central apunte hacia el origen
    if auto_flip:
        cam_pos = c2w[:3, 3]
        center_dir = rays_d[0, H_out//2, W_out//2, :]
        to_origin = -cam_pos
        if torch.dot(center_dir.flatten(), to_origin.flatten()) < 0:
            rays_d = -rays_d

    return rays_o, rays_d  # (1,H,W,3)

class ViewsDataset(Dataset):
    def __init__(self, root: str, size: int = 512, use_depth: bool=False):
        self.items = []
        self.size = size
        self.use_depth = use_depth
        root = Path(root)
        for obj_dir in sorted(root.glob("*")):
            cams_path = obj_dir/"cams.json"
            img_dir, mask_dir, d_dir = obj_dir/"IMG", obj_dir/"MASK", obj_dir/"Depth"
            if not cams_path.exists() or not img_dir.exists():
                continue
            with open(cams_path, "r", encoding="utf-8") as f:
                cams = json.load(f)
            if isinstance(cams, dict) and "cameras" in cams:
                cams = cams["cameras"]
            for i, cam in enumerate(cams):
                name = f"{i:03d}.png" if "file_id" not in cam else f"{cam['file_id']}.png"
                ip = img_dir/name
                mp = mask_dir/name if mask_dir.exists() else None
                dp_npy = (d_dir/f"{Path(name).stem}.npy") if d_dir.exists() else None
                if not ip.exists():
                    continue
                self.items.append((ip, mp, dp_npy, cam))

        if len(self.items) == 0:
            raise RuntimeError(f"No hay datos en {root}")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        ipath, mpath, dpath, cam = self.items[idx]
        im = _pil_open_rgb(str(ipath), self.size)
        img_t = _to_tensor_uint(im)  # (3,H,W)

        mask_t = None
        if mpath is not None and Path(mpath).exists():
            mask_t = _load_mask(str(mpath), self.size)  # (1,H,W)

        depth_t = None
        if self.use_depth and dpath is not None and Path(dpath).exists():
            depth = np.load(str(dpath)).astype(np.float32)
            if depth.max() > 0: depth = depth / (depth.max() + 1e-6)
            depth = np.array(Image.fromarray(depth).resize((self.size, self.size), Image.NEAREST))
            depth_t = torch.from_numpy(depth)[None, ...]  # (1,H,W)

        rays_o, rays_d = _cam_from_json(cam, self.size, self.size)  # (1,H,W,3)
        return {
            "image_pil": im,
            "image": img_t,
            "mask": mask_t,
            "depth": depth_t,
            "rays_o": rays_o,
            "rays_d": rays_d,
        }

# --------------- Pérdidas ---------------
def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]):
    a = _to_chw3(a)
    b = _to_chw3(b)
    if mask is None:
        return F.mse_loss(a, b)
    m = _to_chw1(mask).to(a.device)
    return F.mse_loss(a*m, b*m)

def depth_l1_scale_invariant(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]):
    if gt is None:
        return torch.tensor(0.0, device=pred.device)
    if mask is None:
        mask = torch.ones_like(gt)
    mask = _to_chw1(mask).to(pred.device)
    if pred.dim() == 3 and pred.shape[0] == 1:
        pass  # (1,H,W)
    elif pred.dim() == 2:
        pred = pred.unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(0)
    sel = mask > 0.5
    if sel.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    m_pred = pred[sel].median()
    m_gt   = gt[sel].median()
    scale  = (m_gt / (m_pred + 1e-6)).detach()
    return F.l1_loss(pred*scale, gt)

# --------------- Entrenamiento ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--data_root", type=str, default="data/dataFT")
    ap.add_argument("--save_dir", type=str, default="outputs_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_depth", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--debug_shapes", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    # Añadido 
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model: TSR = TSR.from_pretrained(args.repo_root, config_name="config.yaml", weight_name="model.ckpt")
    print(model)
    model = model.to(device)

    # ---------- Congelamiento-------------------
    print("Congelando el encoder de imagen...")
    for param in model.image_tokenizer.parameters():
        param.requires_grad = False
    for p in model.parameters():
        p.requires_grad_(False)


    # 2. El encoder de imagen en eval (por si tiene BatchNorm, etc.)
    if hasattr(model, "image_tokenizer"):
        model.image_tokenizer.eval()

    # 3. (Opcional pero MUY útil) Gradient checkpointing si existe tokenizer tipo Transformer
    if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "gradient_checkpointing_enable"):
        print("🛡 Activando gradient checkpointing en el tokenizer...")
        model.tokenizer.gradient_checkpointing_enable()

    #replaced = inject_lora_into_model(model.backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    #assert len(replaced) > 0, "No se inyectó LoRA (revisa nombres de capas)."
    
    target_module = model.backbone
    replaced = inject_lora_into_model(
        target_module,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    if len(replaced) == 0:
        raise RuntimeError("❌ No se inyectó ninguna capa LoRA, revisa los nombres de capas.")

    print(f"[LoRA] Capas envueltas ({len(replaced)}):")
    for n in replaced[:12]:
        print("  -", n)

    model.to(device)
    if use_amp:
        print("⚗️ Pasando modelo a float16 (half)...")
        model = model.half()
    # Opcional: Verificar que se congeló correctamente
    # Solo deberían entrenarse los A y B de LoRA
    params = lora_trainable_params(model)
    num_total = sum(p.numel() for p in model.parameters())
    num_train = sum(p.numel() for p in params)

    print(f"📦 Parámetros totales del modelo: {num_total/1e6:.2f} M")
    print(f"🎯 Parámetros entrenables (LoRA): {num_train/1e6:.2f} M")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    ds = ViewsDataset(args.data_root, size=args.img_size, use_depth=args.use_depth)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    collate_fn=lambda b: b, pin_memory=False)

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters(): p.requires_grad_(False)
        _lpips_ok = True
    except Exception:
        lpips_fn = None
        _lpips_ok = False
        print("[AVISO] LPIPS no disponible; continuará sin pérdida perceptual.")

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for batch in dl:
            loss_acc = 0.0
            opt.zero_grad(set_to_none=True)
            # ---- Debuggear -----
            print_cuda_mem("inicio batch")
            # ---- Debuggear -----
            for bi, sample in enumerate(batch):
                print_cuda_mem(f"antes encoder b{bi}")
                # 👇 Todo lo de forward y pérdidas bajo autocast si use_amp=True
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    # Codificar
                    try:
                        scene = model(sample["image_pil"], device=device)
                    except TypeError:
                        scene = model(sample["image_pil"], device)
                    if isinstance(scene, dict):
                        scene_code = scene.get("scene_code") or scene.get("latent") or next(iter(scene.values()))
                    elif isinstance(scene, (list, tuple)):
                        scene_code = scene[0]
                    else:
                        scene_code = scene
                    if torch.is_tensor(scene_code):
                        scene_code = scene_code.to(device)
                    else:
                        raise RuntimeError("TSR.forward no devolvió un tensor compatible para 'scene_code'.")

                    # Rays (BHWC)
                    rays_o = _to_bhwc(sample["rays_o"].to(device))
                    rays_d = _to_bhwc(sample["rays_d"].to(device))

                    if args.debug_shapes and global_step == 0 and bi == 0:
                        print("scene_code", tuple(scene_code.shape))
                        print("rays_o", tuple(rays_o.shape), "rays_d", tuple(rays_d.shape))
                    
                    # ---- Debuggear -----
                    print_cuda_mem(f"antes renderer b{bi}")
                    out = model.renderer(model.decoder, scene_code, rays_o, rays_d)
                    print_cuda_mem(f"después renderer b{bi}")
                    # ---- Debuggear -----

                    if isinstance(out, dict):
                        pred_rgb = out.get("image", out.get("rgb", out.get("color")))
                        pred_depth = out.get("depth", None)
                    else:
                        pred_rgb = out
                        pred_depth = None

                    pred_rgb = _to_chw3(pred_rgb)  # (3,H,W)
                    gt_img   = _to_chw3(sample["image"].to(device))
                    gt_mask  = _to_chw1(sample["mask"].to(device)) if sample["mask"] is not None else None

                    # Pérdidas
                    loss = masked_mse(pred_rgb, gt_img, gt_mask)

                    if _lpips_ok:
                        def _norm(x):
                            x = _to_chw3(x)
                            x = torch.clamp(x,0,1).unsqueeze(0)  # (1,3,H,W)
                            return x*2-1
                        loss = loss + 2.0*lpips_fn(_norm(pred_rgb), _norm(gt_img)).mean()

                    if args.use_depth and sample["depth"] is not None and pred_depth is not None:
                        if pred_depth.dim() == 3 and pred_depth.shape[-1] != 1:
                            # si viene (H,W,1) ya ok; si (H,W) -> (1,H,W)
                            if pred_depth.shape[-1] == 3:
                                # si el renderer devolvió algo en RGB por error, toma canal 0
                                pred_depth = pred_depth[..., 0]
                            pred_depth = pred_depth.unsqueeze(0)
                        loss = loss + 0.1*depth_l1_scale_invariant(pred_depth.to(device), sample["depth"].to(device), gt_mask)
                # cierre de autocast

                # ---- Debuggear -----
                print_cuda_mem(f"antes backward b{bi}")
                # 🔥 backward + scaler
                scaler.scale(loss).backward()
                print_cuda_mem(f"después backward b{bi}")
                # ---- Debuggear -----
                loss_acc += float(loss.detach().cpu())

            # Step de optimizador escaleado
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            # ---- Debuggear -----
            torch.cuda.empty_cache()
            print_cuda_mem("después step")
            # ---- Debuggear -----

            if global_step % 20 == 0:
                print(f"ep {epoch} | step {global_step} | loss {loss_acc/len(batch):.4f}")
            global_step += 1

        save_lora(model, os.path.join(args.save_dir, f"lora_epoch{epoch:03d}.pth"))

    save_lora(model, os.path.join(args.save_dir, "lora_weights.pth"))
    print(f"[OK] Guardado LoRA en {os.path.join(args.save_dir, 'lora_weights.pth')}")

def print_cuda_mem(tag: str):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[MEM {tag}] alloc={alloc:.1f}MB | reserved={reserv:.1f}MB | max={max_alloc:.1f}MB")


if __name__ == "__main__":
    main()
