import os
import json
import random
import cv2
from tqdm import tqdm
import albumentations as A
import numpy as np

# ======================================================================
# --- CONFIGURACIÓN CRÍTICA DE RUTAS Y PARÁMETROS ---
# ⚠️ ADVERTENCIA: Ajusta estas dos rutas si los nombres de las carpetas de Roboflow son diferentes.
# ======================================================================

# 1. RUTA al JSON de Aguacate
COCO_AGUACATE_BASE = 'D:/TODO2/aguacate.v1i.coco/train/_annotations.coco.json' 
# 2. RUTA al JSON de Mango
COCO_MANGO_BASE = 'D:/TODO2/mango.v1i.coco/train/_annotations.coco.json'     

# RUTA RAÍZ: La carpeta que contiene las carpetas de Roboflow
BASE_IMAGE_ROOT = 'D:/TODO2' 
# CARPETA de salida para los nuevos datasets de expertos
OUTPUT_BASE_DIR = 'D:/TODO2/dataset_experts_final' 

# Mapeo de Clases Global (Confirmado por la imagen 61b06f.png)
COCO_CATEGORIES = {
    'aguacate_optimo': 1,
    'aguacate_defecto': 2,
    'mango_optimo': 3,
    'mango_defecto': 4
}
INV_CATEGORIES = {v: k for k, v in COCO_CATEGORIES.items()}

# ======================================================================
# A. Definición de las 5 Funciones de Data Augmentation
# ======================================================================

def get_augmentation_pipeline(aug_index):
    """Devuelve una de las 5 transformaciones robustas para detección de objetos."""
    
    # Albumentations needs Compose with bbox_params to correctly transform bboxes.
    # Build the underlying Compose (with bbox_params for geometric transforms)
    if aug_index == 0:
        base_transform = A.HorizontalFlip(p=1.0)
        compose = A.Compose([base_transform], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    elif aug_index == 1:
        base_transform = A.Affine(translate_percent=0.0, scale=(0.9, 1.1), rotate=10, p=1.0)
        compose = A.Compose([base_transform], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    elif aug_index == 2:
        base_transform = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
        compose = A.Compose([base_transform])
    elif aug_index == 3:
        base_transform = A.GaussNoise(p=1.0)
        compose = A.Compose([base_transform])
    elif aug_index == 4:
        base_transform = A.Blur(blur_limit=3, p=1.0)
        compose = A.Compose([base_transform])
    else:
        return None

    # Return a wrapper that first tries the compose; on bbox-range errors,
    # retry with a tiny PadIfNeeded to give space for rounding overflows.
    def apply_transform(image, bboxes, category_ids=None):
        try:
            if category_ids is not None:
                return compose(image=image, bboxes=bboxes, category_ids=category_ids)
            else:
                return compose(image=image, bboxes=bboxes)
        except Exception as e:
            # If compose has bbox handling, try a padded fallback to avoid tiny >1.0 errors
            try:
                h, w = image.shape[:2]
                # padding: at least 4 pixels, or 2% of the largest dimension
                pad = max(4, int(max(h, w) * 0.02))
                # reconstruct a padded compose: PadIfNeeded + original transforms
                padded_transforms = [A.PadIfNeeded(min_height=h + pad * 2, min_width=w + pad * 2, border_mode=cv2.BORDER_CONSTANT)] + compose.transforms
                if getattr(compose, 'bbox_params', None) is not None:
                    pad_compose = A.Compose(padded_transforms, bbox_params=compose.bbox_params)
                else:
                    pad_compose = A.Compose(padded_transforms)

                if category_ids is not None:
                    return pad_compose(image=image, bboxes=bboxes, category_ids=category_ids)
                else:
                    return pad_compose(image=image, bboxes=bboxes)
            except Exception:
                # re-raise original for outer handler to catch
                raise e

    return apply_transform


# ---- Helpers de bbox -------------------------------------------------
def coco_to_voc_list(bboxes):
    """Convierte lista de bboxes COCO ([x,y,w,h]) a Pascal VOC ([x_min,y_min,x_max,y_max])."""
    voc = []
    for b in bboxes:
        x, y, w, h = b
        voc.append([x, y, x + w, y + h])
    return voc


def voc_to_coco_list(bboxes):
    """Convierte lista de bboxes Pascal VOC ([x_min,y_min,x_max,y_max]) a COCO ([x,y,w,h])."""
    coco = []
    for b in bboxes:
        x_min, y_min, x_max, y_max = b
        coco.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return coco


def denormalize_bboxes_if_needed(bboxes, image_shape):
    """If bboxes appear normalized (0-1), convert them to absolute pixels using image_shape=(h,w,channels)."""
    if not bboxes:
        return bboxes
    # check if coordinates look normalized (all <= 1.01)
    flattened = [c for box in bboxes for c in box]
    max_val = max(flattened)
    if max_val <= 1.01:
        h, w = image_shape[0], image_shape[1]
        denorm = []
        for box in bboxes:
            x_min = box[0] * w
            y_min = box[1] * h
            x_max = box[2] * w
            y_max = box[3] * h
            denorm.append([x_min, y_min, x_max, y_max])
        return denorm
    return bboxes

# ======================================================================
# B. Lógica de Carga y Procesamiento de Expertos
# ======================================================================

def load_and_process_expert(coco_json_path, expert_name, cat_map):
    """Carga los datos COCO de un experto (Aguacate o Mango)."""
    
    if not os.path.exists(coco_json_path):
        raise FileNotFoundError(f"Archivo COCO del Experto {expert_name.upper()} no encontrado en: {coco_json_path}")

    with open(coco_json_path, 'r') as f:
        data = json.load(f)
        
    img_data = {img['id']: img for img in data['images']}
    ann_by_img = {}
    for ann in data['annotations']:
        ann_by_img.setdefault(ann['image_id'], []).append(ann)
        
    expert_categories = [c for c in data['categories'] if c['id'] in cat_map]
    
    return img_data, ann_by_img, expert_categories

def augment_expert_data(img_data, ann_by_img, expert_name, cat_map, expert_categories):
    """Aplica el Data Augmentation (5x) y registra las nuevas anotaciones."""

    current_img_id = 1
    current_ann_id = 1
    
    expert_output = {'coco': {'images': [], 'annotations': [], 'categories': expert_categories}, 'count': 0}
    img_ids_base = list(img_data.keys())
    
    print(f"\nProcesando {len(img_ids_base)} imágenes base para el experto de {expert_name.upper()} (Generando ~{len(img_ids_base) * 6} imágenes)...")

    # 5 Aumentos + 1 Original = 6 versiones por base
    for base_img_id in tqdm(img_ids_base, desc=f"Aumentando {expert_name.upper()}"):
        img_info = img_data[base_img_id]
        base_anns = ann_by_img.get(base_img_id, [])
        
        # --- Carga de imagen y anotaciones ---
        if expert_name == 'aguacate':
            roboflow_json_path = COCO_AGUACATE_BASE
        else:
            roboflow_json_path = COCO_MANGO_BASE

        # El JSON de Roboflow suele estar dentro de la carpeta 'train'.
        # Construimos la ruta usando la carpeta que contiene el JSON (p.ej. .../mango.v1i.coco/train)
        roboflow_dir = os.path.dirname(roboflow_json_path)
        img_path = os.path.join(roboflow_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"\nERROR: No se pudo cargar la imagen en {img_path}. Saltando.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes_coco = [ann['bbox'] for ann in base_anns]
        category_ids = [ann['category_id'] for ann in base_anns]
        
        # Lista de transformaciones (T0 a T5)
        all_images = [(image, bboxes_coco, category_ids)] # T0: Original

        # T1 a T5: Aumentadas
        for aug_index in range(5): 
            transform = get_augmentation_pipeline(aug_index)
            if transform:
                # 1. Convertir COCO (x, y, w, h) a VOC (x_min, y_min, x_max, y_max)
                bboxes_voc = coco_to_voc_list(bboxes_coco)

                # 2. Aplicar transformación (Compose con bbox_params espera bboxes en 'pascal_voc')
                try:
                    augmented = transform(image=image, bboxes=bboxes_voc, category_ids=category_ids)
                except Exception as e:
                    print(f"\nWARNING: Falló la transformación {aug_index} en {img_path}: {e}. Saltando este aumento.")
                    continue

                # 3. Convertir de vuelta a COCO (x, y, w, h)
                # Si albumentations devuelve bboxes normalizados (0..1), denormalizamos primero.
                aug_bboxes_voc = denormalize_bboxes_if_needed(augmented['bboxes'], augmented['image'].shape)
                aug_bboxes_coco = voc_to_coco_list(aug_bboxes_voc)

                all_images.append((augmented['image'], aug_bboxes_coco, augmented['category_ids']))

        
        # 4. Guardar los 6 resultados (Base + 5 Aumentos)
        for i, (aug_image, aug_bboxes_coco, aug_category_ids) in enumerate(all_images):
            
            # Nombre y ruta de salida
            new_img_filename = f"{expert_name}_{base_img_id}_aug{i}.jpg"
            new_img_path_abs = os.path.join(OUTPUT_BASE_DIR, expert_name, new_img_filename)
            
            os.makedirs(os.path.dirname(new_img_path_abs), exist_ok=True)
            cv2.imwrite(new_img_path_abs, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            # Registro en COCO
            new_img_info = {
                'id': current_img_id,
                'file_name': new_img_filename,
                'width': aug_image.shape[1],
                'height': aug_image.shape[0]
            }
            expert_output['coco']['images'].append(new_img_info)

            # Registro de Anotaciones
            for bbox_coco, cat_id in zip(aug_bboxes_coco, aug_category_ids):
                if cat_id in cat_map: 
                    ann = {
                        'id': current_ann_id,
                        'image_id': current_img_id,
                        'category_id': cat_id,
                        'bbox': [round(float(b), 4) for b in bbox_coco],
                        'area': round(bbox_coco[2] * bbox_coco[3], 4),
                        'iscrowd': 0
                    }
                    expert_output['coco']['annotations'].append(ann)
                    current_ann_id += 1
            
            current_img_id += 1
            expert_output['count'] += 1
            
    return expert_output

# ======================================================================
# C. Función de Guardado y División Final (80% Train / 20% Val)
# ======================================================================

def save_coco_files(expert_name, data):
    """Divide los datos aumentados en train/val y guarda los JSON finales."""
    
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, 'annotations'), exist_ok=True)
    coco_data = data['coco']
    
    images = coco_data['images']
    random.shuffle(images)
    split_point = int(0.8 * len(images))
    
    train_images = images[:split_point]
    val_images = images[split_point:]

    ann_by_img_id = {}
    for ann in coco_data['annotations']:
        ann_by_img_id.setdefault(ann['image_id'], []).append(ann)

    def create_split_coco(img_list):
        split_annotations = [ann for img in img_list for ann in ann_by_img_id.get(img['id'], [])]
        return {
            'images': img_list,
            'annotations': split_annotations,
            'categories': coco_data['categories']
        }

    train_coco = create_split_coco(train_images)
    val_coco = create_split_coco(val_images)

    # Guardar JSON
    with open(os.path.join(OUTPUT_BASE_DIR, 'annotations', f'instances_{expert_name}_train.json'), 'w') as f:
        json.dump(train_coco, f)
    
    with open(os.path.join(OUTPUT_BASE_DIR, 'annotations', f'instances_{expert_name}_val.json'), 'w') as f:
        json.dump(val_coco, f)

    print(f"\n--- EXPERTO {expert_name.upper()} FINALIZADO ---")
    print(f"Total de imágenes aumentadas: {data['count']}")
    print(f"Train set: {len(train_coco['images'])} imágenes.")
    print(f"Validation set: {len(val_coco['images'])} imágenes.")


# ======================================================================
# D. Función Principal (Main)
# ======================================================================

def main():
    # Establecer la semilla (seed) para resultados reproducibles
    random.seed(42)
    np.random.seed(42)

    try:
        # --- 1. PROCESAR EXPERTO AGUACATE (IDs 1 y 2) ---
        cat_map_aguacate = {1, 2}
        img_data_a, ann_by_img_a, cat_a = load_and_process_expert(COCO_AGUACATE_BASE, 'aguacate', cat_map_aguacate)
        expert_aguacate_data = augment_expert_data(img_data_a, ann_by_img_a, 'aguacate', cat_map_aguacate, cat_a)
        save_coco_files('aguacate', expert_aguacate_data)
        
        # --- 2. PROCESAR EXPERTO MANGO ---
        # NOTE: Roboflow-exported COCO files use categories with ids 1 and 2 for mango.
        # Use {1,2} here so annotations are kept when filtering.
        cat_map_mango = {1, 2}
        img_data_m, ann_by_img_m, cat_m = load_and_process_expert(COCO_MANGO_BASE, 'mango', cat_map_mango)
        expert_mango_data = augment_expert_data(img_data_m, ann_by_img_m, 'mango', cat_map_mango, cat_m)
        save_coco_files('mango', expert_mango_data)
        
        print("\n🎉 ¡Proceso de datasets de Modelos de Experto completado!")
        print(f"Los datasets aumentados y divididos están listos en: {OUTPUT_BASE_DIR}")
        
    except FileNotFoundError as e:
        print(f"ERROR CRÍTICO: Asegúrate de que los archivos COCO y las rutas sean correctas. {e}")
        print("Verifica especialmente que los nombres de las carpetas de Roboflow ('aguacate.v1.i.coco' y 'mango.v1.i.coco') sean correctos.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el aumento: {e}")

if __name__ == '__main__':
    main()