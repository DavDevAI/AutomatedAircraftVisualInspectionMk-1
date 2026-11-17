# AircraftDetectorMk-1: Inspección Visual Automatizada de la Piel de Aeronaves

## Introducción
Este repositorio contiene la primera iteración (Mark 1) del detector de defectos superficiales en la piel de fuselaje de aeronaves, desarrollado como parte de un Trabajo Profesional de Ingeniería Aeronáutica en Mantenimiento. El objetivo general del trabajo es implementar un modelo automatizado de inspección visual mediante el entrenamiento de una red neuronal convolucional, con apoyo de YOLO y OpenCV, para eficientar las tareas de inspección visual y reducir la componente de error humano.

El alcance de Mark 1 comprende el entrenamiento de un modelo YOLOv8 de una sola clase (`defect`), los dos scripts de inferencia local y la actualización menor aplicada al script detector antes de la segunda iteración. La segunda iteración del proyecto queda fuera de este repositorio.

## Dataset Utilizado
El modelo se entrenó con un único conjunto de datos público, sin combinarlo con otros.

### 1. aircraft-skin-defects-new-dataset (v2, "size 640 stock defect class")
- **Descripción**: Contiene 1,115 imágenes a color de defectos en piel de aeronave, anotadas con recuadros delimitadores en formato YOLOv8. Esta versión fusiona todas las anotaciones en una sola clase, `defect`. El reparto es de 962 imágenes de entrenamiento, 102 de validación y 51 de prueba.
- **Preprocesamiento**: Redimensionado a 640 × 640 píxeles por estiramiento (*stretch*). No se aplicó aumento de datos en la exportación.
- **Acceso**: Disponible en [Roboflow Universe](https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset), con licencia CC BY 4.0. Se descargó en formato YOLOv8 y se alojó en Google Drive.
- **Uso**: Las imágenes se utilizaron tal como se exportaron. La partición de validación se empleó durante el entrenamiento para medir cada época y seleccionar `best.pt`.
- **Citación**: Dibya Dillip. (2023). *aircraft-skin-defects-new-dataset* [Conjunto de datos]. Roboflow Universe.

### Configuración (`data.yaml`)
```yaml
path: ../drive/MyDrive/Datasets/AircraftSkinDefects
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['defect']
```

## Herramientas y Referencias de Base
A continuación, se presentan las herramientas y la referencia documental sobre las que se construyó esta iteración:

### 1. Ultralytics YOLOv8
- **Descripción**: Implementación oficial de YOLOv8, un detector de objetos de una sola etapa. Se usó la variante *large* (`yolov8l.pt`), preentrenada en COCO, como base de transferencia de aprendizaje.
- **Relevancia**: Provee el entrenamiento por línea de comandos y la interfaz de inferencia que usan los scripts.

### 2. OpenCV y cvzone
- **Descripción**: OpenCV (`cv2`) se emplea para leer imágenes y video y desplegar resultados. cvzone se emplea para trazar los recuadros y las etiquetas sobre la imagen.
- **Relevancia**: Constituyen la capa de visión por computadora de los scripts de inferencia.

### 3. Google Colaboratory
- **Descripción**: Entorno de notebooks en la nube con GPU NVIDIA Tesla T4 en su capa gratuita.
- **Relevancia**: Fue el entorno de entrenamiento, dado que no se contó con GPU dedicada en el equipo local.

### 4. Using Convolutional Neural Networks to Automate Aircraft Maintenance Visual Inspection
- **Descripción**: Artículo de Doğru et al. (2020) publicado en [MDPI Aerospace](https://www.mdpi.com/2226-4310/7/12/171), que aplica Mask R-CNN a la detección de abolladuras en aeronaves.
- **Relevancia**: Es el antecedente documental del trabajo. En este proyecto se optó por la detección de una sola etapa con YOLOv8 en lugar de Mask R-CNN.

## Implementación de la Primera Iteración
### Título: Detección de Defectos en Piel de Fuselaje con YOLOv8 (una clase)
### Pasos Principales
1. **Configuración del Entorno de Entrenamiento**:
   - Se creó un notebook en Google Colab con acelerador GPU y se verificó su asignación con `!nvidia-smi`.
   - Se instaló la librería con `!pip install ultralytics`.
   - Se montó Google Drive con `drive.mount('/content/drive')` y se ajustaron las rutas del `data.yaml`.

2. **Entrenamiento**:
   - Se ejecutó el siguiente comando:
     ```
     !yolo task=detect mode=train model=yolov8l.pt data=/content/drive/MyDrive/Datasets/AircraftSkinDefects/data.yaml epochs=50 imgsz=640
     ```
   - Los resultados se generaron en `/content/runs/detect/train`.

3. **Respaldo y Exportación de Pesos**:
   - El almacenamiento local de Colab es temporal y se elimina al cerrarse la sesión. Por ello se incorporó el respaldo a Drive con `!cp -r /content/runs /content/drive/MyDrive/AircraftSkinDefects/runs_backup/`.
   - El archivo `best.pt` se descargó al equipo local y se colocó en la carpeta `Weights/`.

4. **Inferencia Local**:
   - `AircraftDefectDetector.py`: detección sobre imágenes estáticas, con umbral de confianza de 0.35.
   - `AircraftDefectDetectorLive.py`: detección sobre video (`Media/AircraftDefect.mp4`), con `stream=True` y umbral de confianza de 0.4.

5. **Actualización del Script Detector (previa a la segunda iteración)**:
   - **Problema detectado**: el script rotulaba las detecciones con una lista escrita a mano, `class_labels = ['dent', 'scratch', 'crack', 'corrosion']`. Como el modelo solo conoce la clase de índice 0, toda detección se mostraba como `dent`.
   - **Corrección**: se eliminó la lista y los nombres se leen del propio modelo con `yolo_model.names[cls]`, por lo que el rótulo mostrado es `defect`.
   - **Adición**: la imagen procesada se guarda como evidencia con `cv2.imwrite` en `Media/Resultados/`.
   - En `AircraftDefectDetectorLive.py` el cambio es equivalente: se elimina `classNames` y se usa `model.names[cls]`.

### Estructura del Repositorio
```
AircraftDetectorMk-1/
├── Weights/
│   └── best.pt
├── Media/
│   ├── Defects/
│   ├── Flush/
│   ├── Resultados/
│   └── AircraftDefect.mp4
├── AircraftDefectDetector.py
└── AircraftDefectDetectorLive.py
```

### Código del Detector
A continuación, se presenta el script `AircraftDefectDetector.py` con la actualización aplicada:

```python
import cv2
import math
import cvzone
from ultralytics import YOLO

# Carga el modelo YOLO (los nombres de clase vienen dentro de los pesos)
yolo_model = YOLO("Weights/best.pt")
print(yolo_model.names)  # {0: 'defect'}

# Carga la imagen
image_path = "Media/Flush/pos1.jpg"
img = cv2.imread(image_path)

# Realiza la detección
results = yolo_model(img)

# Dibuja bounding boxes
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.35:  # Umbral de confianza
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{yolo_model.names[cls]} {conf}', (x1, y1 - 10),
                               scale=0.8, thickness=1, colorR=(255, 0, 0))

# Guarda la evidencia y muestra el resultado
cv2.imwrite("Media/Resultados/pos1_pred.jpg", img)
cv2.imshow("Image", img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)
```

### Consideraciones Técnicas
- **Entorno**: Python con las librerías `ultralytics`, `opencv-python` y `cvzone`. Instalación: `pip install ultralytics opencv-python cvzone`.
- **Ejecución**: `python AircraftDefectDetector.py`. La ventana de resultado se cierra con la tecla `q`.
- **Carpeta de salida**: `Media/Resultados/` debe existir antes de ejecutar el script, porque `cv2.imwrite` no la crea.
- **Hardware**: El entrenamiento requirió la GPU de Colab. La inferencia se ejecuta en el equipo local.

## Resultados de la Primera Iteración
Las métricas se obtuvieron sobre la partición de validación (102 imágenes, 148 instancias anotadas), a partir de la matriz de confusión generada por Ultralytics (umbral de confianza 0.25, IoU 0.45).

- **Aprendizaje**: Las pérdidas de entrenamiento y validación descendieron y las métricas aumentaron durante las 50 épocas, sin señales de sobreajuste.
- **Convergencia**: El modelo no convergió. En la época 50 las pérdidas seguían descendiendo y la mAP seguía aumentando.
- **Prueba cualitativa**: El script se probó con fotografías propias de la piel de un CRJ-200, ajenas al entrenamiento. Unas se identificaron correctamente como sin defecto y otras como con defecto, y un tubo pitot no se marcó como defecto.

## Limitaciones y Alcances Futuros
Las limitaciones identificadas en Mark 1 son las siguientes:
- **Una sola clase**: El modelo localiza defectos, pero no distingue su tipo.
- **Entrenamiento sin convergencia**: 50 épocas no fueron suficientes, y la mAP@0.5 quedó por debajo de la meta de 0.85 planteada en el trabajo.
- **Métricas deducidas**: Los valores se derivaron de la matriz de confusión sobre validación. No se ejecutó una validación sobre la partición de prueba.
- **Persistencia**: Los resultados se escribieron en el disco temporal de Colab y dependieron de un respaldo manual.

Como alcances futuros se consideran:
- **Clasificación por tipo de defecto**: Entrenamiento con clases diferenciadas.
- **Lógica de decisión**: Condicionales de profundidad y tolerancia, y criterio go/no-go.
- **Pruebas en entorno real**: Pruebas en plataforma y con inspector técnico.
- **Despliegue**: Integración del modelo en una aplicación móvil.

## Tabla Resumen del Dataset

| **Dataset**                              | **Número de Imágenes** | **Clases**     | **Reparto (train / valid / test)** | **Acceso**        | **Preprocesamiento**                    |
|------------------------------------------|------------------------|----------------|------------------------------------|-------------------|-----------------------------------------|
| aircraft-skin-defects-new-dataset (v2)   | 1,115                  | 1 (`defect`)   | 962 / 102 / 51                     | Roboflow Universe | Estiramiento a 640 × 640, sin aumento   |

## Tabla de Resultados

| **Métrica**                    | **Expresión**            | **Valor**         |
|--------------------------------|--------------------------|-------------------|
| Verdaderos positivos (TP)      | —                        | 87                |
| Falsos positivos (FP)          | —                        | 52                |
| Falsos negativos (FN)          | —                        | 61                |
| Precisión                      | TP / (TP + FP) = 87/139  | 0.626             |
| Exhaustividad (recall)         | TP / (TP + FN) = 87/148  | 0.588             |
| F1-score                       | 2TP / (2TP + FP + FN) = 174/287 | 0.606      |
| mAP@0.5 (`best.pt`)            | —                        | ≈ 0.57            |
| mAP@0.5:0.95 (`best.pt`)       | —                        | ≈ 0.28            |

Los valores de mAP son aproximados y corresponden a la lectura de la gráfica `results.png`.

## Tabla Comparativa de Scripts

| **Script**                        | **Entrada**                  | **Umbral de Confianza** | **Salida**                                         |
|-----------------------------------|------------------------------|-------------------------|----------------------------------------------------|
| AircraftDefectDetector.py         | Imagen estática              | 0.35                    | Ventana con detecciones e imagen guardada en `Media/Resultados/` |
| AircraftDefectDetectorLive.py     | Video (`AircraftDefect.mp4`) | 0.4                     | Ventana con detecciones cuadro a cuadro            |

## Conclusión
**Mark 1** demuestra la factibilidad del enfoque: el flujo completo de datos, entrenamiento en Colab e inferencia local funciona, y el modelo aprende a localizar defectos en la piel de fuselaje. Con precisión y exhaustividad cercanas a 0.6 y una mAP@0.5 de aproximadamente 0.57, el resultado constituye una línea base y no un desempeño final. La actualización del script detector deja la inferencia coherente con el modelo de una clase y cierra el alcance de este repositorio.

**Referencias**:
- [aircraft-skin-defects-new-dataset (Dibya Dillip, 2023)](https://universe.roboflow.com/dibya-dillip/aircraft-skin-defects-new-dataset)
- [Ultralytics YOLOv8 (Jocher, Chaurasia y Qiu, 2023)](https://github.com/ultralytics/ultralytics)
- [OpenCV (Bradski, 2000)](https://opencv.org/)
- [cvzone](https://github.com/cvzone/cvzone)
- [Google Colaboratory](https://colab.research.google.com/)
- [Using Convolutional Neural Networks to Automate Aircraft Maintenance Visual Inspection (Doğru et al., 2020)](https://www.mdpi.com/2226-4310/7/12/171)