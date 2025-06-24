# src/visualization_utils.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import random
from PIL import Image
from .data_utils import load_image, augment_random

def plot_image_grid(df: pd.DataFrame, root_dir: str, n_rows: int = 5, n_cols: int = 5, figsize=(20, 20)):
    """
    Muestra una grilla de imágenes aleatorias desde el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene la información de las imágenes.
        root_dir (str): La ruta raíz donde se encuentran las carpetas o se usa image_path.
        n_rows (int): Número de filas en la grilla.
        n_cols (int): Número de columnas en la grilla.
        figsize (tuple): Tamaño de la figura de matplotlib.
    """
    total_images = n_rows * n_cols
    num_samples = min(total_images, len(df))
    if num_samples == 0:
        print("No hay imágenes para mostrar.")
        return
    random_indices = random.sample(range(len(df)), num_samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        row = df.iloc[idx]
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = row['image_path']
        else:
            img_path = os.path.join(root_dir, row['relative_path'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            axes[i].axis('off')
            continue
        axes[i].imshow(img)
        axes[i].set_title(str(row.get('class', '')).replace('___', '\n'), fontsize=12)
        axes[i].axis('off')

    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_full_hierarchical_distribution(df: pd.DataFrame, normalize_y_axis: bool = False):
    """
    Crea una visualización de tres paneles con la distribución jerárquica completa
    y añade una leyenda de colores en los dos paneles inferiores.
    Permite normalizar el eje Y para mostrar porcentajes.
    """
    print("Generando gráfico de distribución jerárquica de 3 niveles...")

    group_counts = df['group'].value_counts(normalize=normalize_y_axis).sort_index()
    palette = plt.get_cmap('tab20')(np.linspace(0, 1, len(group_counts)))
    color_map = {group: palette[i] for i, group in enumerate(group_counts.index)}

    tag_counts = df['tag'].value_counts(normalize=normalize_y_axis)
    tag_to_group_map = df[['tag', 'group']].drop_duplicates().set_index('tag')['group'].to_dict()
    sorted_tags = sorted(tag_counts.index, key=lambda tag: (tag_to_group_map.get(tag, ''), tag))
    tag_counts_sorted = tag_counts.reindex(sorted_tags)
    tag_colors = [color_map.get(tag_to_group_map.get(tag), (0.5,0.5,0.5,1)) for tag in tag_counts_sorted.index]

    class_counts = df['class'].value_counts(normalize=normalize_y_axis).sort_index()
    class_colors = [color_map.get(name.split('___')[0], (0.5,0.5,0.5,1)) for name in class_counts.index]
    legend_patches = [mpatches.Patch(color=color, label=group) for group, color in color_map.items()]

    fig, axes = plt.subplots(3, 1, figsize=(20, 32), gridspec_kw={'height_ratios': [1, 2, 3]})
    fig.suptitle('Análisis de Distribución Jerárquica del Dataset', fontsize=22, y=1.02)
    y_label = 'Proporción (%)' if normalize_y_axis else 'Número de Imágenes'

    sns.barplot(x=group_counts.index, y=group_counts.values * (100 if normalize_y_axis else 1),
                ax=axes[0], palette=[color_map[g] for g in group_counts.index])
    axes[0].set_title('Nivel 1: Distribución por Grupo', fontsize=16)
    axes[0].set_ylabel(y_label, fontsize=12)
    axes[0].tick_params(axis='x', rotation=30, labelsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), ha='right')

    sns.barplot(x=tag_counts_sorted.index, y=tag_counts_sorted.values * (100 if normalize_y_axis else 1),
                ax=axes[1], palette=tag_colors)
    axes[1].set_title('Nivel 2: Distribución por Tag', fontsize=16)
    axes[1].set_ylabel(y_label, fontsize=14)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), ha='right')
    axes[1].legend(handles=legend_patches, title='Grupo', loc='best', fontsize='medium')

    sns.barplot(x=class_counts.index, y=class_counts.values * (100 if normalize_y_axis else 1),
                ax=axes[2], palette=class_colors)
    axes[2].set_title('Nivel 3: Distribución por Clase', fontsize=16)
    axes[2].set_ylabel(y_label, fontsize=14)
    axes[2].tick_params(axis='x', rotation=90, labelsize=10)
    axes[2].legend(handles=legend_patches, title='Grupo', loc='best', fontsize='small')

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

def pie_graph(df: pd.DataFrame,
              crop_col: str,
              class_col: str,
              ncols: int = 3):
    """
    Analiza un DataFrame y genera una grilla de gráficos de torta, uno por cada
    cultivo, mostrando la distribución de sus clases.
    Versión con gráficos más grandes y leyenda en la parte inferior.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        crop_col (str): Nombre de la columna que identifica al cultivo.
        class_col (str): Nombre de la columna que identifica la clase.
        ncols (int): Número de columnas para la grilla.
    """
    print(f"Generando distribución de '{class_col}' por '{crop_col}'...")

    unique_crops = sorted(df[crop_col].unique())
    num_crops = len(unique_crops)
    nrows = (num_crops + ncols - 1) // ncols
    
    # --- CAMBIO 1: Gráficos más grandes ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 8)) 
    fig.suptitle('Distribución de Clases por Cultivo', fontsize=20, weight='bold')
    axes = axes.flatten()

    for i, crop_name in enumerate(unique_crops):
        ax = axes[i]
        crop_df = df[df[crop_col] == crop_name]
        class_distribution = crop_df[class_col].value_counts()
        legend_labels = [str(label).split('___')[-1].replace('_', ' ') for label in class_distribution.index]

        wedges, _ = ax.pie(
            class_distribution,
            startangle=90,
            colors=plt.cm.Paired.colors
        )

        ax.set_title(str(crop_name).replace("_", " ").title(), fontsize=16)
        ax.axis('equal')

        # --- CAMBIO 2: Leyenda en la parte inferior ---
        ax.legend(wedges, legend_labels,
                  title="Clases",
                  loc="best",
                  bbox_to_anchor=(1, 0, 0.5, 1), 
                  fontsize=12)

    for j in range(num_crops, len(axes)):
        axes[j].axis('off')

    # Ajustamos el layout para evitar solapamientos
    plt.tight_layout() 
    plt.show()

def plot_spectral_signatures(hist_df: pd.DataFrame, title_prefix: str = "Firma Espectral"):
    """
    Grafica las firmas espectrales (histogramas RGB promedio) desde un DataFrame pre-calculado.

    Args:
        hist_df (pd.DataFrame): DataFrame con las columnas ['category', 'bin', 'r', 'g', 'b'].
        title_prefix (str): Prefijo para el título de cada gráfico.
    """
    
    categories = hist_df["category"].unique()
    
    for category in categories:
        plt.figure(figsize=(12, 6))
        
        # Filtramos los datos para la categoría actual
        category_data = hist_df[hist_df["category"] == category]
        
        # Graficamos cada canal de color
        plt.plot(category_data["bin"], category_data["r"], color='red', label='Rojo', alpha=0.8)
        plt.plot(category_data["bin"], category_data["g"], color='green', label='Verde', alpha=0.8)
        plt.plot(category_data["bin"], category_data["b"], color='blue', label='Azul', alpha=0.8)
        
        plt.title(f"{title_prefix} - {category}", fontsize=16)
        plt.xlabel("Intensidad de Píxel (0-255)")
        plt.ylabel("Frecuencia Promedio Normalizada")
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

def plot_distribution(df: pd.DataFrame, column: str, hue_column: str = None, title: str = None, normalize: bool = False, rotation: int = 45):
    """
    Plots the distribution of a specified column, optionally with a hue and normalization.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to plot the distribution for.
        hue_column (str, optional): An optional column to use for hue. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Distribution of {column}".
        normalize (bool): If True, normalize the counts to percentages. Defaults to False.
        rotation (int): Rotation of x-axis labels. Defaults to 45.
    """
    plt.figure(figsize=(18, 8))
    
    if normalize:
        # Calculate value counts normalized
        counts = df.groupby(hue_column)[column].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
        sns.barplot(data=counts, x=column, y='percentage', hue=hue_column, palette='viridis')
        plt.ylabel('Proporción de Imágenes (%)')
    else:
        sns.countplot(data=df, x=column, hue=hue_column, palette='viridis')
        plt.ylabel('Número de Imágenes')

    plt.xticks(rotation=rotation)
    plt.title(title if title else f'Distribution of {column}' + (f' by {hue_column}' if hue_column else ''))
    plt.xlabel(column)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(dfs: list, labels: list, title: str):
    """
    Grafica la distribución de clases para una o más DataFrames para comparación.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Preparar un DataFrame combinado para graficar con Seaborn
    combined_df_list = []
    for i, df in enumerate(dfs):
        temp_df = df['class'].value_counts().reset_index()
        temp_df.columns = ['class', 'count']
        temp_df['source'] = labels[i]
        combined_df_list.append(temp_df)
    
    full_df = pd.concat(combined_df_list).sort_values('count', ascending=False)
    
    sns.barplot(data=full_df, x='count', y='class', hue='source', ax=ax, orient='h')
    
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Número de Muestras', fontsize=12)
    ax.set_ylabel('Clase', fontsize=12)
    ax.legend(title='Estrategia')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.show()

def augment_random_plot(df: pd.DataFrame, num_calls: int = 4, num_steps: int = 3):
    """
    Toma una imagen aleatoria y la aumenta varias veces con augment_random
    para mostrar su variabilidad y las transformaciones aplicadas.
    """
    sample = df.sample(1).iloc[0]
    img_path = sample.get('image_path')
    class_name = sample['class'].replace('___', '\n')

    if not img_path or not os.path.exists(img_path):
        print(f"La ruta de la imagen no es válida o no existe: {img_path}")
        return

    print(f"Mostrando ejemplos de 'augment_random' para la clase: {class_name.replace(chr(10), ' ')}")
    
    fig, axes = plt.subplots(1, num_calls + 1, figsize=(22, 5))
    fig.suptitle(f"Variabilidad de augment_random (con num_steps={num_steps})", fontsize=16, y=1.05)

    original_img = Image.open(img_path).convert('RGB')
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for i in range(num_calls):
        # AHORA RECIBIMOS LA IMAGEN Y LA LISTA DE NOMBRES
        augmented_img, transform_names = augment_random(img_path, num_steps=num_steps)
        
        if augmented_img:
            axes[i + 1].imshow(augmented_img)
            # CONSTRUIMOS EL TÍTULO CON LOS NOMBRES
            title = " +\n".join(transform_names) if transform_names else "Sin Aumento"
            axes[i + 1].set_title(title)
        else:
            axes[i + 1].set_title(f"Error en Llamada #{i + 1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_augmentation_comparison(df: pd.DataFrame, image_root_dir: str, num_samples: int = 3):
    """
    Muestra una comparación visual clara entre aumentos conservadores e intensivos.
    """
    sample_df = df.sample(min(num_samples, len(df)), random_state=42)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1: axes = [axes]

    fig.suptitle('Comparación de Estrategias de Aumentación', fontsize=20, y=1.02)

    for i, (_, row) in enumerate(sample_df.iterrows()):
        original_image = load_image(df, row.name, image_root_dir)
        img_path = row.get('image_path')
        
        if original_image is None:
            for j in range(3): axes[i, j].axis('off')
            continue
        
        # Estrategia Conservadora
        fixed_augs_list = augment_image(img_path)
        chosen_tuple = random.choice(fixed_augs_list) if fixed_augs_list else (None, None)
        conservative_img, title_text = (chosen_tuple[1], f"Conservador\n({chosen_tuple[0]})") if chosen_tuple else (None, "Error")

        # Estrategia Intensiva
        intensive_img = augment_random(img_path, num_steps=4)
        
        axes[i, 0].imshow(original_image); axes[i, 0].set_title("Original", fontsize=14)
        axes[i, 1].imshow(conservative_img or original_image); axes[i, 1].set_title(title_text, fontsize=14)
        axes[i, 2].imshow(intensive_img or original_image); axes[i, 2].set_title("Intensivo\n(4 pasos)", fontsize=14)
        
        class_name = row['class'].replace('___', '\n')
        axes[i, 0].set_ylabel(class_name, fontsize=12, weight='bold', rotation=0, labelpad=50, ha='right', va='center')

        for j in range(3): axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_class_distribution(dfs: list, labels: list, title: str):
    """Grafica la distribución de clases para una o más DataFrames para comparación."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))
    combined_df_list = []
    for i, df in enumerate(dfs):
        temp_df = df['class'].value_counts().reset_index(); temp_df.columns = ['class', 'count']; temp_df['source'] = labels[i]
        combined_df_list.append(temp_df)
    full_df = pd.concat(combined_df_list).sort_values('count', ascending=False)
    sns.barplot(data=full_df, x='count', y='class', hue='source', ax=ax, orient='h')
    ax.set_title(title, fontsize=16, weight='bold'); ax.set_xlabel('Número de Muestras'); ax.set_ylabel('Clase'); ax.legend(title='Estrategia'); ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.show()

print("Todas las funciones han sido definidas y corregidas.")
