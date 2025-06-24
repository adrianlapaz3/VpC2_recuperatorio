# src.data_utils.py

import os
import pandas as pd
import logging
from PIL import Image, ImageFilter
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import seaborn as sns   

# ----------------------------------
# Utils de procesamiento de imágenes
# ----------------------------------

def process_image_directory(root_directory: str, folder_separator: str = '___') -> pd.DataFrame:
    """
    Recorre un directorio raíz, extrae rutas de imágenes y metadatos de subcarpetas.
    Se asume que subcarpetas tienen nombre 'group___class'.
    Retorna DataFrame con columnas: relative_path, class, group, tag.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_data = []
    if not os.path.isdir(root_directory):
        logging.error(f"El directorio raíz especificado no existe: {root_directory}")
        return pd.DataFrame()
    logging.info(f"Procesando directorio de imágenes: {root_directory}")
    for subdirectory_name in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, subdirectory_name)
        if not os.path.isdir(subdirectory_path):
            continue
        if folder_separator not in subdirectory_name:
            logging.warning(f"'{subdirectory_name}' no contiene el separador '{folder_separator}'. Saltando.")
            continue
        try:
            group_name, tag_name = subdirectory_name.split(folder_separator, 1)
        except ValueError:
            logging.warning(f"No se pudo dividir '{subdirectory_name}'. Saltando.")
            continue
        for filename in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, filename)
            if not os.path.isfile(file_path):
                continue
            _, file_extension = os.path.splitext(filename)
            if file_extension.lower() in IMAGE_EXTENSIONS:
                image_data.append({
                    'relative_path': os.path.join(subdirectory_name, filename),
                    'class': subdirectory_name,
                    'group': group_name,
                    'tag': tag_name
                })
    if not image_data:
        logging.warning("No se encontraron imágenes válidas en el directorio.")
    df = pd.DataFrame(image_data)
    return df


def load_image(df: pd.DataFrame, index: int, root_dir: str):
    """Carga una imagen PIL desde una fila específica de un DataFrame."""
    try:
        row = df.iloc[index]
        full_path = os.path.join(root_dir, row['relative_path']) if 'relative_path' in row else row.get('image_path')
        img = Image.open(full_path).convert('RGB')
        return img
    except FileNotFoundError:
        print(f"Archivo no encontrado: {full_path}")
        return None
    except Exception as e:
        print(f"Error al cargar la imagen en el índice {index}: {e}")
        return None


def calculate_average_histograms_by_category(df: pd.DataFrame, root_dir: str, category_col: str, bins: int = 256, max_images_per_category: int = 50):
    """Calcula el histograma RGB promedio para cada categoría única."""
    def _load(full_path):
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            return None
    grouped = df.groupby(category_col)
    histograms_data = defaultdict(lambda: {'r': [], 'g': [], 'b': [], 'count': 0})
    print(f"Calculando histogramas promedio por '{category_col}'...")
    for category_name, group_df in grouped:
        image_count = 0
        for path in group_df.get("full_path", []):
            if max_images_per_category and image_count >= max_images_per_category:
                break
            img = _load(path)
            if img:
                arr = np.array(img)
                hr, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 256))
                hg, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 256))
                hb, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 256))
                histograms_data[category_name]['r'].append(hr)
                histograms_data[category_name]['g'].append(hg)
                histograms_data[category_name]['b'].append(hb)
                image_count += 1
        histograms_data[category_name]['count'] = image_count
        print(f"  - Categoría '{category_name}': {image_count} imágenes procesadas.")
    result_list = []
    for category, hists in histograms_data.items():
        if hists['count'] > 0:
            avg_r = np.mean(hists['r'], axis=0)
            avg_g = np.mean(hists['g'], axis=0)
            avg_b = np.mean(hists['b'], axis=0)
            for i in range(bins):
                result_list.append({
                    "category": category,
                    "bin": i,
                    "r": avg_r[i],
                    "g": avg_g[i],
                    "b": avg_b[i],
                })
    return pd.DataFrame(result_list)


def split_data(df: pd.DataFrame,
               target_column: str = 'class',
               test_size: float = 0.2,
               validation_size: float = 0.0,
               random_state: int = 42,
               split_column_name: str = 'split') -> pd.DataFrame:
    """Añade una columna indicando la división (train/valid/test) usando estratificación."""
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el DataFrame.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe ser un float > 0 y < 1.")
    if not (0.0 <= validation_size < 1.0):
        raise ValueError("validation_size debe ser un float >= 0 y < 1.")
    if test_size + validation_size >= 1.0:
        raise ValueError("La suma de test_size y validation_size debe ser menor que 1.0.")
    df[split_column_name] = 'unassigned'
    labels = df[target_column]
    indices = df.index
    remaining_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    df.loc[test_indices, split_column_name] = 'test'
    if validation_size > 0:
        relative_val_size = validation_size / (1.0 - test_size)
        remaining_labels = df.loc[remaining_indices, target_column]
        train_indices, validation_indices = train_test_split(
            remaining_indices,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=remaining_labels
        )
        df.loc[train_indices, split_column_name] = 'train'
        df.loc[validation_indices, split_column_name] = 'valid'
    else:
        df.loc[remaining_indices, split_column_name] = 'train'
    train_prop = (df[split_column_name] == 'train').mean()
    valid_prop = (df[split_column_name] == 'valid').mean()
    test_prop = (df[split_column_name] == 'test').mean()
    print(f"División completada y estratificada por '{target_column}':")
    print(f"  - Train:      {train_prop:.1%}")
    if valid_prop > 0:
        print(f"  - Validation: {valid_prop:.1%}")
    print(f"  - Test:       {test_prop:.1%}")
    return df

# ----------------------------------
# Transformaciones para augmentations
# ----------------------------------
# --- SECCIÓN 1: DEFINICIÓN DE TRANSFORMACIONES DE AUMENTO (VERSIÓN CORREGIDA) ---
FINAL_IMG_SIZE = (224, 224)

# 1. Definimos las transformaciones una por una
zoom_crop_transform = T.RandomResizedCrop(size=FINAL_IMG_SIZE, scale=(0.5, 1.0))
random_rotation = T.RandomRotation(degrees=45, fill=128)
color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
gaussian_blur = T.GaussianBlur(kernel_size=(7, 13), sigma=(1.0, 3.0))

def addnoise(input_image: Image.Image, noise_factor: float = 0.1) -> Image.Image:
    inputs = T.ToTensor()(input_image)
    noisy = inputs + torch.rand_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0, 1.)
    return T.ToPILImage()(noisy)

# Creamos la función lambda para el ruido por separado
noise_transform = lambda img: addnoise(img, random.uniform(0.05, 0.20))


# 2. Creamos la lista de transformaciones que usará augment_random
RANDOM_TRANSFORMS_LIST = [
    zoom_crop_transform,
    random_rotation,
    color_jitter,
    gaussian_blur,
    noise_transform  # Usamos la variable de la función lambda
]

# 3. Creamos el diccionario de nombres que corresponde EXACTAMENTE a la lista
TRANSFORM_NAMES = {
    zoom_crop_transform: "Zoom/Recorte",
    random_rotation: "Rotación",
    color_jitter: "Jitter Color",
    gaussian_blur: "Desenfoque",
    noise_transform: "Ruido"  # Ahora la llave es el objeto lambda, no un string
}

# 4. Actualizamos augment_random para que la búsqueda de nombres sea directa
def augment_random(img_path: str, num_steps: int = 2) -> tuple:
    """
    Aplica un número aleatorio de transformaciones aleatorias a una imagen.
    Devuelve una tupla con (imagen_final, lista_de_nombres_de_transformaciones).
    """
    try:
        img = T.Resize((256, 256))(Image.open(Path(img_path)).convert('RGB'))
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en augment_random: {img_path}")
        return None, []
        
    steps_to_apply = random.randint(1, num_steps)
    applied_transforms = []

    for _ in range(steps_to_apply):
        transform_func = random.choice(RANDOM_TRANSFORMS_LIST)
        img = transform_func(img)
        
        # Búsqueda de nombres ahora es directa y sin errores
        transform_name = TRANSFORM_NAMES.get(transform_func, "Desconocido")
        applied_transforms.append(transform_name)
        
    final_img = T.Resize(FINAL_IMG_SIZE)(img)
    return final_img, applied_transforms

# --- SECCIÓN 2: FUNCIONES DE ESTRATEGIAS DE BALANCEO ---

# EN ARCHIVO: src/data_utils.py

def apply_strategy_1_multiplicative(df_train: pd.DataFrame, df_full_counts: pd.Series, save_aug_dir: str) -> pd.DataFrame:
    """ESTRATEGIA 1: Aumenta datos con un multiplicador basado en el CONTEO GLOBAL."""
    out_records = []
    base_aug_dir = Path(save_aug_dir)
    print("Iniciando Estrategia 1 [Multiplicativa con Umbrales Globales]...")
    
    for cls, train_cnt in df_train['class'].value_counts().items():
        full_cnt = df_full_counts.get(cls, 0)
        print(f"  - Clase '{cls}': {train_cnt} en train (Conteo global: {full_cnt})")
        
        cls_df = df_train[df_train['class'] == cls].copy()
        paths = cls_df['image_path'].tolist()
        
        for _, row in cls_df.iterrows():
            out_records.append({'image_path': row['image_path'], 'class': row['class']})
        
        if full_cnt > 4000: mult = 1
        elif full_cnt > 2000: mult = 2
        elif full_cnt > 500: mult = 3
        else: mult = 6
        
        if mult > 1:
            save_dir = base_aug_dir / cls
            save_dir.mkdir(parents=True, exist_ok=True)
            needed = train_cnt * (mult - 1)
            print(f"    -> Conteo global ({full_cnt}) en rango. Aplicando aumento x{mult}. Se generarán {needed} imágenes.")
            i = 0
            while needed > 0:
                img_path = paths[i % len(paths)]
                
                # --- LÍNEA MODIFICADA AQUÍ ---
                aug_img, _ = augment_random(img_path, num_steps=2)
                
                if aug_img:
                    fname = f"{Path(img_path).stem}_aug_strat1_{i}.png"
                    out_path = save_dir / fname
                    aug_img.save(out_path) # Ahora aug_img es una imagen, no una tupla
                    out_records.append({'image_path': str(out_path), 'class': cls})
                    needed -= 1
                i += 1
    print("Estrategia 1 completada.")
    return pd.DataFrame(out_records)

# EN ARCHIVO: src/data_utils.py

def apply_strategy_2_balanced_final(df_train: pd.DataFrame, df_full_counts: pd.Series, save_aug_dir: str, target_count: int = 2000) -> pd.DataFrame:
    """ESTRATEGIA 2: Balanceo total a 2000, usando reglas de over y under-sampling."""
    temp_records = []
    base_aug_dir = Path(save_aug_dir)
    print(f"Iniciando Estrategia 2 [Etapa 1: Over-sampling Granular]...")
    for cls, train_cnt in df_train['class'].value_counts().items():
        full_cnt = df_full_counts.get(cls, 0)
        print(f"  - Clase '{cls}': {train_cnt} en train (Conteo global: {full_cnt})")
        cls_df = df_train[df_train['class'] == cls].copy()
        paths = cls_df['image_path'].tolist()
        
        if full_cnt <= 150: mult = 20
        elif full_cnt <= 250: mult = 10
        elif full_cnt <= 300: mult = 6
        elif full_cnt <= 450: mult = 5
        elif full_cnt <= 550: mult = 4
        elif full_cnt <= 1000: mult = 2
        elif full_cnt <= 3000: mult = 2
        else: mult = 1
            
        for _, row in cls_df.iterrows():
            temp_records.append({'image_path': row['image_path'], 'class': row['class'], 'is_original': True})
        
        if mult > 1:
            save_dir = base_aug_dir / cls
            save_dir.mkdir(parents=True, exist_ok=True)
            needed = train_cnt * (mult - 1)
            print(f"    -> Aumentando x{mult}. Se generarán {needed} nuevas imágenes.")
            i = 0
            while needed > 0:
                img_path = paths[i % len(paths)]
                num_steps = 4 if full_cnt < 250 else 2
                
                # --- LÍNEA MODIFICADA AQUÍ ---
                aug_img, _ = augment_random(img_path, num_steps=num_steps)
                
                if aug_img:
                    fname = f"{Path(img_path).stem}_aug_strat2_{i}.png"
                    out_path = save_dir / fname
                    aug_img.save(out_path) # Ahora aug_img es una imagen, no una tupla
                    temp_records.append({'image_path': str(out_path), 'class': cls, 'is_original': False})
                    needed -= 1
                i += 1
    
    df_augmented = pd.DataFrame(temp_records)
    
    print("\nIniciando Estrategia 2 [Etapa 2: Under-sampling a 2000]...")
    final_records = []
    for cls in df_augmented['class'].unique():
        cls_df_aug = df_augmented[df_augmented['class'] == cls]
        if len(cls_df_aug) > target_count:
            print(f"  - Clase '{cls}': Recortando de {len(cls_df_aug)} a {target_count} muestras.")
            final_records.append(cls_df_aug.sample(n=target_count, random_state=42))
        else:
            final_records.append(cls_df_aug)
            
    df_final = pd.concat(final_records).reset_index(drop=True)
    print("Estrategia 2 completada.")
    return df_final

