# Clasificación de enfermedades en plantas con algortimos de visión por computador 

Este proyecto explora y compara diversas técnicas de Deep Learning para la clasificación de imágenes de hojas de plantas, con el objetivo de identificar 38 categorías distintas que incluyen diferentes especies y enfermedades. Se pone un énfasis especial en el manejo de datos desbalanceados a través de distintas estrategias de aumento de datos y en la comparación de arquitecturas de modelos (entrenamiento desde cero, Transfer Learning y Fine-Tuning).

![image](https://github.com/user-attachments/assets/327649b6-a0ea-4e7c-b541-d92f7ac65148)

![image](https://github.com/user-attachments/assets/b76ad6ac-0280-41ad-a935-03dbda9535a3)

![image](https://github.com/user-attachments/assets/ceb90e9d-ca55-4b79-9739-1b601c7d6dc4)


## 📝 Resumen del Proyecto
El objetivo principal es construir un clasificador de imágenes robusto para identificar enfermedades en plantas. Para lograrlo, se sigue un flujo de trabajo sistemático:
1.  **Análisis Exploratorio de Datos (EDA):** Se analiza la distribución de clases para identificar el desbalanceo inherente en el dataset.
2.  **Aumento de Datos:** Se implementan y comparan dos estrategias de oversampling para mitigar el desbalanceo de clases.
3.  **Modelado y Entrenamiento:** Se entrenan tres tipos de modelos sobre cada estrategia de datos:
    * **Baseline:** Una red convolucional entrenada desde cero.
    * **Transfer Learning:** Usando una arquitectura pre-entrenada (ResNet18) con las capas base congeladas.
    * **Fine-Tuning:** Adaptando las capas superiores de la arquitectura pre-entrenada para especializarla en el dominio de las plantas.
4.  **Evaluación:** Se comparan los 6 modelos resultantes en un conjunto de prueba no visto para determinar la mejor combinación de estrategia de datos y técnica de entrenamiento.

## 💾 Dataset
El proyecto utiliza el dataset PlantVillage, un conjunto de datos público con más de 54,000 imágenes a color que cubren 38 clases distintas de enfermedades y especies de plantas. El dataset original se encuentra en Kaggle (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data), pero para este proyecto se descarga una copia directamente desde Google Drive en el notebook 1_EDA_split.ipynb.

## 📂 Estructura del Repositorio
El flujo de trabajo está organizado en una serie de notebooks de Jupyter, diseñados para ser ejecutados en secuencia.

```
.
├── notebooks/
│   ├── 1_EDA_split.ipynb           # 1. Análisis exploratorio y división de datos
│   ├── 2_data_augmentation.ipynb   # 2. Implementación de estrategias de aumento
│   ├── 3_baseline-TransfL.ipynb    # 3. Entrenamiento de modelos Baseline y Transfer Learning
│   ├── 4_fine_tunning.ipynb        # 4. Entrenamiento de modelos con Fine-Tuning
│   └── 5_evaluate_models.ipynb     # 5. Evaluación final y comparación de todos los modelos
├── src/
│   ├── architecture.py             # Define la arquitectura del modelo PyTorch
│   └── data_utils.py               # Funciones para el manejo de datos y augmentation
├── data/
│   ├── raw/                        # Contiene el dataset original descomprimido
│   └── processed/                  # Contiene los DataFrames y datos aumentados
└── README.md                       # Este archivo
```

## ⚙️ Metodología

### 1. Análisis y División de Datos (`1_EDA_split.ipynb`)
Se realiza un análisis inicial para visualizar la distribución de imágenes por clase. Se evidencia un fuerte desbalanceo. Posteriormente, el dataset completo se divide de forma estratificada en conjuntos de **entrenamiento (70%)**, **validación (20%)** y **prueba (10%)** para asegurar que todas las clases estén representadas en cada división.

### 2. Estrategias de Aumento de Datos (`2_data_augmentation.ipynb`)
Para combatir el desbalanceo, se exploran dos estrategias de oversampling sobre el conjunto de entrenamiento:
* **Estrategia 1 (Multiplicativa):** Aumenta las clases minoritarias con un multiplicador basado en su conteo global. Las clases más raras reciben un aumento mayor (hasta x6), mientras que las más comunes no se modifican. No se realiza ningún recorte de las clases dominantes.
* **Estrategia 2 (Balanceo):** Intenta crear un dataset más balanceado, aumentando las clases minoritarias y recortando las mayoritarias a un umbral máximo de muestras.

### 3. Modelado y Entrenamiento (`3_baseline-TransfL.ipynb` y `4_fine_tunning.ipynb`)
Se definen tres enfoques de entrenamiento que se aplican a los datos de ambas estrategias:
* **Baseline:** Un modelo convolucional entrenado desde cero para aprender características específicas del dominio.
* **Transfer Learning:** Se utiliza una arquitectura pre-entrenada en ImageNet. Se congelan todas las capas convolucionales y solo se entrena un nuevo clasificador añadido al final.
* **Fine-Tuning:** Se parte del modelo de Transfer Learning, pero se "descongelan" las últimas capas convolucionales y se re-entrenan con una tasa de aprendizaje muy baja, permitiendo que el modelo adapte sus características de alto nivel al dataset de plantas.

## 📊 Resultados y Conclusiones

La evaluación final se realizó sobre el conjunto de prueba, que el modelo nunca vio durante el entrenamiento.

### Conclusiones Principales
1.  **La Mejor Combinación:** La **Estrategia 1 (aumento multiplicativo)** combinada con el modelo de **Fine-Tuning** demostró ser la más efectiva, alcanzando la mayor precisión (96.8%) y el mejor F1-Score macro (0.959).

2.  **Fine-Tuning es Superior:** El Fine-Tuning superó consistentemente tanto al Baseline como al Transfer Learning simple. Esto demuestra el poder de aprovechar el conocimiento pre-entrenado y, a la vez, especializarlo en el problema concreto.

3.  **El Peligro del "Sobre-Balanceo":** La Estrategia 2, que buscaba un mayor balance recortando clases dominantes, obtuvo peores resultados que la Estrategia 1. Esto sugiere que las muestras de las clases mayoritarias contenían información valiosa y que un aumento más moderado y enfocado (Estrategia 1) es más beneficioso.

4.  **Transferencia Negativa:** En la Estrategia 1, el modelo Baseline superó al de Transfer Learning simple. Esto es un claro indicador de "transferencia negativa", donde las características de ImageNet no se adaptaban bien al dominio de las plantas y el modelo que aprendió desde cero logró mejores resultados. El Fine-Tuning fue clave para corregir este efecto.

## 🚀 Cómo Ejecutar el Proyecto

### 1. Requisitos
Asegúrate de tener un entorno con Python 3.8+ y las siguientes librerías. Puedes instalarlas usando pip:
```bash
pip install torch torchvision pandas numpy scikit-learn seaborn matplotlib gdown tqdm
```

### 2. Secuencia de Ejecución
Para reproducir los resultados, los notebooks deben ejecutarse en el siguiente orden:

1.  **`notebooks/1_EDA_split.ipynb`**: Para descargar, analizar y dividir el dataset.
2.  **`notebooks/2_data_augmentation.ipynb`**: Para generar los datasets aumentados según las dos estrategias.
3.  **`notebooks/3_baseline-TransfL.ipynb`**: Para entrenar los modelos Baseline y de Transfer Learning.
4.  **`notebooks/4_fine_tunning.ipynb`**: Para entrenar los modelos de Fine-Tuning.
5.  **`notebooks/5_evaluate_models.ipynb`**: Para cargar todos los modelos guardados, evaluarlos en el set de prueba y generar los gráficos y reportes finales.
