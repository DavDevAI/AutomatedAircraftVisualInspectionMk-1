# Proyecto de Inspección Visual Automatizada de Aeronaves

## Introducción
La inspección visual automatizada de aeronaves es un área clave en el mantenimiento predictivo aeronáutico, ya que permite detectar defectos como abolladuras, grietas o corrosión de manera eficiente, reduciendo errores humanos y costos operativos. Este informe detalla datasets públicos y proyectos relevantes para desarrollar un proyecto de fin de materia basado en redes neuronales convolucionales (CNN), utilizando tus conocimientos en OpenCV y librerías como Keras y TensorFlow. El objetivo es proponer un proyecto acotado, implementable en un entorno de escritorio (Jupyter o VS Code), y que sea completable en un plazo razonable, similar a un proyecto full-stack en programación web.

## Datasets Públicos para Inspección Visual de Aeronaves
La disponibilidad de datasets grandes y abiertos es crucial para entrenar modelos CNN. A continuación, se presentan los datasets más relevantes identificados:

### 1. Aircraft_Fuselage_DET2023 Dataset
- **Descripción**: Este dataset contiene 5,601 imágenes de defectos en fuselajes de aeronaves, clasificados en cuatro tipos (por ejemplo, abolladuras, grietas, corrosión). Las imágenes fueron capturadas en diferentes condiciones de iluminación, lo que lo hace robusto para aplicaciones de visión por computadora.
- **Acceso**: Disponible en [IEEE DataPort](https://ieee-dataport.org/documents/aircraftfuselagedet2023-aircraft-fuselage-defect-detection-dataset). Puedes descargar un archivo ZIP que incluye carpetas con etiquetas en formatos COCO, VOC y YOLO, además de imágenes sin etiquetar.
- **Uso**: Ideal para entrenar modelos CNN debido a su tamaño y especificidad. Puedes usar OpenCV para preprocesar las imágenes (por ejemplo, normalización, aumento de datos) y Keras/TensorFlow para entrenar un modelo como YOLOv5 o U-Net.
- **Citación**: Si usas este dataset, cita el artículo asociado: "A Semi-Supervised Aircraft Fuselage Defect Detection Network with Dynamic Attention and Class-aware Adaptive Pseudo-Label Assignment".

### 2. aircraft-skin-defects-new-dataset
- **Descripción**: Contiene 1,115 imágenes de defectos en la piel de aeronaves, alojado en [Roboflow Universe](https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset). Es más pequeño que el dataset anterior, pero sigue siendo útil para pruebas iniciales o proyectos con menos recursos computacionales.
- **Acceso**: Regístrate en Roboflow para descargar el dataset o usar su API. Las instrucciones específicas están disponibles en la plataforma tras iniciar sesión.
- **Uso**: Adecuado para experimentos iniciales o como complemento al dataset de IEEE DataPort. Puedes aplicar técnicas de aumento de datos con OpenCV para compensar su menor tamaño.

### 3. Otros Datasets
- **Dataset de "Visual inspection of aircraft skin"**: Incluye 276 imágenes de defectos en la piel de aeronaves (resolución 960 × 720 píxeles), pero no está públicamente disponible. Puedes intentar contactar a los autores del artículo ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1000936122000814)) para solicitar acceso.
- **DAGM Dataset**: Aunque no es específico para aeronaves, contiene 1,150 imágenes de defectos en superficies (1,000 sin defectos y 150 con defectos por clase). Puede ser útil para pruebas iniciales, pero no es ideal para tu enfoque aeronáutico.
- **FGVC Aircraft Dataset** ([Kaggle](https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft)): Contiene 10,000 imágenes para clasificación de aeronaves, pero no está enfocado en defectos, por lo que no es directamente aplicable.

### Generación de Datos Sintéticos
Si los datasets disponibles no son suficientes, puedes generar datos sintéticos usando herramientas como Blender o Unity, como se describe en el artículo "Artificial Dataset Generation for Automated Aircraft Visual Inspection" ([IEEE Xplore](https://ieeexplore.ieee.org/document/9696375)). Este enfoque crea imágenes renderizadas con anotaciones automáticas, útil para aumentar el tamaño del dataset, aunque requiere tiempo para configurar el entorno de generación.

## Proyectos y Artículos de Referencia
A continuación, se presentan proyectos y artículos que utilizan CNN para la detección de defectos en aeronaves, los cuales puedes usar como base para tu proyecto:

### 1. Using Convolutional Neural Networks to Automate Aircraft Maintenance Visual Inspection
- **Descripción**: Este artículo ([MDPI](https://www.mdpi.com/2226-4310/7/12/171)) utiliza MASK R-CNN para detectar abolladuras en aeronaves. Explora técnicas como aumento de datos (volteo, rotación, desenfoque) y un clasificador previo para mejorar el rendimiento.
- **Dataset**: Usa un dataset personalizado de 56 imágenes con abolladuras y 49 sin defectos, recolectado en el hangar de Abu Dhabi Polytechnic y fuentes en línea. No está públicamente disponible, pero la metodología es replicable con datasets como Aircraft_Fuselage_DET2023.
- **Metodología**: Combina MASK R-CNN con aumento de datos y transferencia de aprendizaje desde un modelo de detección de abolladuras en autos. Usa validación cruzada de 10 pliegues debido a la escasez de datos.
- **Resultados**: La combinación de MASK R-CNN y aumento de datos mejora la precisión en la detección y segmentación de defectos.
- **Relevancia**: Puedes adaptar su enfoque de MASK R-CNN y usar OpenCV para preprocesar imágenes del dataset de IEEE DataPort.

### 2. GitHub: xaviergoby/Deep-Learning-and-Computer-Vision-for-Structural-Crack-Detection-And-Classification
- **Descripción**: Este repositorio ([GitHub](https://github.com/xaviergoby/Deep-Learning-and-Computer-Vision-for-Structural-Crack-Detection-And-Classification)) desarrolla un modelo CNN para clasificar la longitud de grietas en materiales aeroespaciales, incorporando conocimiento del dominio de ingeniería aeroespacial.
- **Dataset**: Usa datos de pruebas de fatiga en aluminio de grado aeroespacial, complementados con CIFAR-10 y MNIST para comparaciones. No es público, pero puedes replicar el enfoque con el dataset de IEEE DataPort.
- **Metodología**: Compara aprendizaje sesgado (biased) y no sesgado, utilizando una CNN para clasificar grietas según la curva de propagación de fatiga.
- **Resultados**: El aprendizaje sesgado logra un 97.55% de precisión en la validación, mientras que el no sesgado alcanza un 99.45% pero selecciona parámetros no deseados (como intensidad de luz solar).
- **Relevancia**: Este proyecto es ideal para inspirar tu enfoque, ya que se centra en defectos estructurales en aeronaves y usa CNN, alineándose con tus herramientas.

### 3. A Deep-Learning-Based Approach for Aircraft Engine Defect Detection
- **Descripción**: Este artículo ([MDPI](https://www.mdpi.com/2075-1702/11/2/192)) propone un marco de deep learning para inspección de motores mediante borescopes, utilizando U-Net para detectar defectos en palas de compresores de alta presión.
- **Dataset**: No especifica un dataset público, pero aborda problemas de desenfoque por movimiento con técnicas de visión por computadora y GAN.
- **Metodología**: Combina U-Net con un método híbrido de eliminación de desenfoque y aborda problemas de desbalance de datos y defectos pequeños.
- **Resultados**: Mejora la detección de defectos en imágenes de borescope, con alta precisión en entornos ruidosos.
- **Relevancia**: Aunque se centra en motores, su enfoque en U-Net y preprocesamiento con visión por computadora es útil para tu proyecto.

### 4. Deep learning-based defects detection of certain aero-engine blades and vanes with DDSC-YOLOv5s
- **Descripción**: Este artículo ([Scientific Reports](https://www.nature.com/articles/s41598-022-17340-7)) introduce una red YOLOv5s modificada con redes convolucionales deformables para detectar defectos en palas y vanos de motores.
- **Dataset**: No menciona un dataset público, pero su enfoque en defectos de motores es relevante.
- **Metodología**: Optimiza YOLOv5s para detectar defectos con formas variadas, mejorando la adaptabilidad de las características.
- **Resultados**: Logra alta precisión en la detección en tiempo real de defectos en motores.
- **Relevancia**: Puedes implementar una arquitectura YOLOv5 similar para detectar defectos en fuselajes usando el dataset de IEEE DataPort.

### 5. Deep Learning-Based Real Time Defect Detection for Optimization of Aircraft Manufacturing and Control Performance
- **Descripción**: Este artículo ([MDPI](https://www.mdpi.com/2504-446X/7/1/31)) propone un marco de deep learning en tiempo real para identificar componentes defectuosos en la fabricación de aeronaves.
- **Dataset**: No especifica un dataset público, pero se centra en defectos en procesos de fabricación.
- **Metodología**: Usa CNN para clasificar estados anormales en componentes intermedios.
- **Resultados**: Reduce la probabilidad de retrabajo al detectar defectos antes de la integración final.
- **Relevancia**: Su enfoque en tiempo real es útil para un proyecto práctico y escalable.

## Propuesta de Proyecto
### Título: Detección Automatizada de Defectos en Fuselajes de Aeronaves con CNN
### Pasos Principales
1. **Selección de Dataset**:
   - Usa el **Aircraft_Fuselage_DET2023 Dataset** (5,601 imágenes) como base principal debido a su tamaño y relevancia.
   - Opcionalmente, complementa con el **aircraft-skin-defects-new-dataset** (1,115 imágenes) para pruebas adicionales.

2. **Preprocesamiento**:
   - Usa OpenCV para normalizar imágenes, aplicar aumento de datos (volteo, rotación, cambio de brillo) y manejar variaciones en iluminación.
   - Convierte las imágenes a un formato adecuado para CNN (por ejemplo, escala de grises o RGB normalizado).

3. **Modelado**:
   - **Modelo Baseline**: Implementa un modelo simple como una CNN básica o una red densa para clasificar imágenes como defectuosas o no defectuosas.
   - **Modelo CNN Avanzado**: Usa una arquitectura como YOLOv5 o U-Net, inspirándote en los artículos mencionados. Por ejemplo, adapta el enfoque de MASK R-CNN del artículo de MDPI o la YOLOv5s modificada de Scientific Reports.
   - Entrena el modelo con Keras/TensorFlow en Jupyter o VS Code.

4. **Evaluación**:
   - Usa métricas como precisión, recall, F1-score y el coeficiente de Dice para evaluar la detección y segmentación de defectos.
   - Compara el rendimiento del modelo baseline con el modelo CNN avanzado.

5. **Análisis y Conclusión**:
   - Discute cómo los resultados pueden aplicarse al mantenimiento aeronáutico, como la reducción de tiempos de inspección.
   - Propón mejoras, como incorporar datos sintéticos o explorar arquitecturas híbridas (por ejemplo, CNN-LSTM).

### Código de Ejemplo
A continuación, se proporciona un ejemplo básico de código en Python para preprocesar imágenes con OpenCV y entrenar una CNN con Keras/TensorFlow:

```python
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Cargar y preprocesar imágenes
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalización
    return img

# Ejemplo de carga de datos (ajusta según tu dataset)
images = []  # Lista de imágenes preprocesadas
labels = []  # Etiquetas (0: sin defecto, 1: con defecto)

# Modelo CNN simple
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Entrenar modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

### Consideraciones Técnicas
- **Entorno**: Configura un entorno de Python con OpenCV, Keras, TensorFlow y Scikit-learn. Usa TensorFlow 2.x para compatibilidad con CNN.
- **Hardware**: Un equipo con GPU es ideal para acelerar el entrenamiento, pero un CPU es suficiente para datasets de tamaño moderado.
- **Tiempo Estimado**: El proyecto puede completarse en unas pocas semanas, dependiendo de tu experiencia. Es comparable a un proyecto full-stack en términos de complejidad.
- **Escalabilidad**: Puedes expandir el proyecto explorando arquitecturas más avanzadas (como YOLOv8) o integrando datos sintéticos.

## Alternativas y Otras Áreas
Si prefieres explorar otras áreas del mantenimiento aeronáutico, considera:
- **Inspección de Motores con Borescopes**: Usa datasets de imágenes de borescope (como los mencionados en el artículo de MDPI) para detectar defectos internos en motores.
- **Detección de Defectos en Materiales Compuestos**: Aplica CNN para identificar defectos en materiales compuestos, como se describe en "Deep CNN-Based Automated Optical Inspection for Aerospace Components" ([ERAU](https://commons.erau.edu/edt/757)).
- **Generación de Datos Sintéticos**: Si los datasets son limitados, implementa un pipeline de generación de datos sintéticos con Blender, siguiendo el artículo de IEEE Xplore.

## Tabla Comparativa de Datasets

| **Dataset**                          | **Número de Imágenes** | **Tipos de Defectos** | **Acceso**                     | **Ventajas**                              | **Desafíos**                              |
|--------------------------------------|------------------------|-----------------------|--------------------------------|-------------------------------------------|-------------------------------------------|
| Aircraft_Fuselage_DET2023            | 5,601                 | 4 tipos               | IEEE DataPort                  | Gran tamaño, etiquetas en múltiples formatos | Requiere preprocesamiento extenso         |
| aircraft-skin-defects-new-dataset     | 1,115                 | No especificado       | Roboflow Universe              | Fácil acceso, adecuado para pruebas iniciales | Menor tamaño, menos diversidad            |
| Visual inspection of aircraft skin    | 276                   | No especificado       | No público                     | Alta resolución, específico para piel     | No disponible públicamente                |
| DAGM Dataset                         | 1,150                 | 6 clases              | Público                        | Útil para pruebas generales               | No específico para aeronaves              |

## Tabla Comparativa de Proyectos

| **Proyecto/Artículo**                | **Arquitectura CNN** | **Dataset**                     | **Resultados**                          | **Relevancia para el Proyecto**          |
|--------------------------------------|----------------------|---------------------------------|-----------------------------------------|------------------------------------------|
| Using CNN to Automate Aircraft Inspection | MASK R-CNN          | Personalizado (105 imágenes)    | Mejora en detección con aumento de datos | Metodología replicable con datasets públicos |
| xaviergoby GitHub                    | CNN personalizada    | Pruebas de fatiga aeroespacial | 97.55% precisión en grietas             | Enfoque en defectos estructurales         |
| Aircraft Engine Defect Detection      | U-Net, GAN           | No público                      | Alta precisión en borescopes            | Útil para preprocesamiento avanzado       |
| DDSC-YOLOv5s                         | YOLOv5s modificada   | No público                      | Detección en tiempo real                | Arquitectura adaptable a fuselajes        |

## Conclusión
El **Aircraft_Fuselage_DET2023 Dataset** es la mejor opción para tu proyecto debido a su tamaño y relevancia. Puedes basar tu implementación en el repositorio de **xaviergoby** para clasificar defectos o en el artículo de MDPI para detección con MASK R-CNN. Usa OpenCV para preprocesamiento y Keras/TensorFlow para entrenar una CNN, asegurando un proyecto técnico, práctico y completable en un plazo razonable. Si necesitas más datos, considera generar imágenes sintéticas con Blender. Este enfoque te permitirá desarrollar habilidades en visión por computadora y aprendizaje profundo, alineándose con tus objetivos académicos.

**Referencias**:
- [Aircraft_Fuselage_DET2023 Dataset](https://ieee-dataport.org/documents/aircraftfuselagedet2023-aircraft-fuselage-defect-detection-dataset)
- [aircraft-skin-defects-new-dataset](https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset)
- [Using Convolutional Neural Networks to Automate Aircraft Maintenance Visual Inspection](https://www.mdpi.com/2226-4310/7/12/171)
- [xaviergoby/Deep-Learning-and-Computer-Vision-for-Structural-Crack-Detection-And-Classification](https://github.com/xaviergoby/Deep-Learning-and-Computer-Vision-for-Structural-Crack-Detection-And-Classification)
- [A Deep-Learning-Based Approach for Aircraft Engine Defect Detection](https://www.mdpi.com/2075-1702/11/2/192)
- [Deep learning-based defects detection of certain aero-engine blades and vanes with DDSC-YOLOv5s](https://www.nature.com/articles/s41598-022-17340-7)
- [Artificial Dataset Generation for Automated Aircraft Visual Inspection](https://ieeexplore.ieee.org/document/9696375)