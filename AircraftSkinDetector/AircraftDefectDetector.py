import cv2
import math
import cvzone
from ultralytics import YOLO

# Carga el modelo YOLO
yolo_model = YOLO("Weights/best.pt")

# Define las etiquetas de clases (ajusta según data.yaml de Roboflow)
class_labels = ['dent', 'scratch', 'crack', 'corrosion']  # Ejemplo; verifica en data.yaml
#para configurar junto con el data.yaml para los umbrales y tolerancias y definir con mejor inteligencia que tipo de defecto es 
#si es scratch crack or corrosion

#SERÍA PONER EN ALCANCES LA CONDICIONALES DE PROFUNDIDAD Y TOLERANCIA ASÍ COMO DAR LOS UMBRALES Y DARLE MAS INTELIGENCIA A LA DETECCION DETECTANDO MAS DEFECTOS

#INCLUSO AHONDAR EN CIENCIA DE MATERIALES PARA INCLUSO CLASIFICAR DEFECTOS PUNTUALES NO SOLO EN PIEL SINO ESTRUCTURA Y ALABES DE ROTOR/MOTOR/ETC


# Carga la imagen
image_path = "Media/Flush/pos1.jpg"
img = cv2.imread(image_path)

# Realiza la detección
results = yolo_model(img)

# Dibuja bounding boxes
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        if conf > 0.35:  # Umbral de confianza
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf}', (x1, y1 - 10),
                               scale=0.8, thickness=1, colorR=(255, 0, 0)) #CHECAR DIMENSIONES DE THICKNESS Y PROFUNDIDADES
#IGYUAL AÑADIR LOGICA QUE DE SI NO ES UN GO


# Muestra el resultado
cv2.imshow("Image", img)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cv2.waitKey(1)