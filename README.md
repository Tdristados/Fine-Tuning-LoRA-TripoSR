# Fine Tuning LoRA aplicado a TripoSR

<a href="https://huggingface.co/stabilityai/TripoSR"><img src="https://img.shields.io/badge/🔗%20TripoSR-HuggingFace-orange"></a>
<a href="https://arxiv.org/pdf/2403.02151"><img src="https://img.shields.io/badge/📄%20Paper%20TripoSR-ArXiv-B31B1B"></a>
<a href="/mnt/data/CVProjectPresent.pdf"><img src="https://img.shields.io/badge/📘%20Informe%20del%20Proyecto-PDF-blue"></a>

---

## 📌 Descripción

Este proyecto desarrolla un **Fine-Tuning con LoRA sobre el modelo TripoSR** para mejorar la reconstrucción 3D a partir de imágenes específicas del dataset utilizado.  
El objetivo fue **especializar el modelo en ciertas formas, texturas o configuraciones**, obteniendo mejores *scene codes* y posteriormente mallas `.obj` adaptadas al dominio del proyecto.

### ¿Qué se modificó exactamente con LoRA?

El entrenamiento LoRA se aplicó sobre:

- **El backbone** del encoder de imagen de TripoSR.  
- Los **módulos de atención**, específicamente los valores **K, V y C** dentro de las capas *self-attention* y *cross-attention* del modelo.

Esto permite que el modelo aprenda nuevas variaciones de forma **sin modificar todos los pesos base**, manteniendo estable la arquitectura principal.

Durante el fine-tuning se inyectaron los módulos LoRA en:

- `transformer_blocks[i].attn1.to_q`
- `transformer_blocks[i].attn1.to_k`
- `transformer_blocks[i].attn1.to_v`
- `transformer_blocks[i].attn1.to_out.0`
- `transformer_blocks[i].attn2.to_q`
- `transformer_blocks[i].attn2.to_k`
- `transformer_blocks[i].attn2.to_v`
- `transformer_blocks[i].attn2.to_out.0`

(Ver lógica en `fine_tuning.py` del proyecto).

---

## ⚙️ Detalles técnicos  
### Cambios realizados al modelo TripoSR original

La única modificación directa hecha al código fuente de TripoSR ocurrió en:

```bash
tsr/models/nerf_renderer.py
```

### ¿Qué se cambió?

Se añadió un parámetro **`chunk_size`** para controlar la cantidad de rayos procesados en cada iteración del renderizado NeRF.

### ¿Por qué es necesario?

TripoSR extrae la malla usando **Marching Cubes**, que requiere muestrear la función SDF sobre un grid 3D. Esto puede consumir **enormes cantidades de VRAM**.  
El `chunk_size` permite dividir este procesamiento en bloques más pequeños para:

- evitar *OOM errors* (out of memory),
- permitir el entrenamiento en GPUs de 24–48 GB de VRAM,
- mejorar la estabilidad del entrenamiento LoRA,
- mantener el *rendering pipeline* sin romper la arquitectura interna.

Esta modificación **no altera la arquitectura del modelo**, solo la eficiencia computacional.

---

## 🛠️ Instalación y configuración del entorno

### ⚠️ Requisitos mínimos de hardware
| Recurso | Valor mínimo |
|--------|--------------|
| GPU | **CUDA 11.x**, ideal 11.4 |
| VRAM | **≥ 25 GB** (para fine-tuning) |
| RAM | 16 GB |
| Sistema | Linux, CentOS7, Ubuntu 20.04 |
| Python | 3.9.x |

---

## 📥 Instalación paso a paso (Conda + CUDA + TorchMCubes)

### 1️⃣ Crear entorno conda  
*(El comando exacto depende del usuario)*

```bash
conda create -n TripoSR python=3.9
conda activate TripoSR
```

---

### 2️⃣ Instalar PyTorch compatible con CUDA 11.8

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

---

### 3️⃣ Instalar dependencias del proyecto

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Fijar NumPy a la versión correcta

```bash
pip install "numpy==1.26.4" --force-reinstall
python -c "import numpy as np; print('numpy', np.__version__)"
```

---

### 5️⃣ Instalar TorchMCubes

```bash
pip install --no-build-isolation "torchmcubes @ git+https://github.com/tatsy/torchmcubes.git"
```

Si falla:

```bash
pip install --upgrade pip
unzip torchmcubes
cd torchmcubes
pip install -e .
```

---

### 6️⃣ Instalar herramientas para compilación

```bash
pip install -U pip setuptools wheel
pip install -U scikit-build-core ninja cmake
```

---

### 7️⃣ Cargar módulos del clúster (HPC)

```bash
module purge all
module load gnu9
module load cuda/11.4
```

---

### 8️⃣ Validación final

```bash
python -c "import torch, torchmcubes, numpy as np; print('CUDA available?', torch.cuda.is_available()); print('ok torchmcubes, numpy', np.__version__)"
```

---

## 🚀 Uso del proyecto

A continuación se explica cómo ejecutar **fine_tuning.py** y luego **inference_lora.py**.

---

# 🎯 Fine Tuning (Entrenamiento LoRA)

Archivo: **fine_tuning.py**

### Ejecución general:

```bash
python fine_tuning.py     --input_folder ruta/a/imagenes/     --output_dir outputs_lora/     --batch_size 1     --lr 1e-4     --epochs 100     --r 16     --alpha 16     --dropout 0.05
```

### Parámetros importantes:

| Parámetro | Significado |
|----------|-------------|
| `input_folder` | Carpeta con imágenes PNG/JPG |
| `output_dir` | Donde se guardarán los pesos LoRA |
| `r` | Dimensión interna de LoRA |
| `alpha` | Escala de LoRA |
| `dropout` | Dropout aplicado a módulos LoRA |
| `epochs` | Iteraciones de entrenamiento |
| `lr` | Learning rate |
| `device` | CPU o GPU |

**Salida esperada:**

```bash
outputs_lora/lora_weights.pth
```

---

# 🔍 Inferencia con LoRA

Archivo: **inference_lora.py**

### Ejecución:

```bash
python inference_lora.py     --image_path examples/chair.png     --lora_path outputs_lora/lora_weights.pth     --repo_root .
```

### ¿Qué hace el script?

1. Carga el modelo base TripoSR (`model.ckpt`)
2. Inyecta los módulos LoRA (**igual que en el entrenamiento**)
3. Carga los pesos entrenados
4. Procesa la imagen → genera el `scene_code`
5. Usa Marching Cubes para extraer la malla
6. Exporta:

```bash
mesh_lora.obj
```

---

## 📦 Estructura del proyecto

```bash
TripoSR-LORA/
├── fine_tuning.py
├── inference_lora.py
├── tsr/
│   └── models/
│       └── nerf_renderer.py   # Modificado con chunk_size
├── requirements.txt
├── outputs_lora/
│   └── lora_weights.pth
└── README.md
```

---

## 👨‍💻 Autoría

Proyecto desarrollado como parte del **Proyecto Final de Computer Vision**, Mauricio Andrés Manrique — Universidad del Rosario (2025).
