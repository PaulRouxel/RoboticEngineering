import cv2
import numpy as np
#from ultralytics import 

# Chargement du modèle YOLO pré-entraîné
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Charger les classes (coco.names contient les noms des classes)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Liste pour stocker les chemins des images
image_paths = ["frigo1.png", "frigo2.png", "frigo3.png", "frigo4.png", "frigo5.png",
               "frigo6.png", "frigo7.png", "frigo8.png", "frigo9.png", "frigo10.png"]

# Lire et traiter les 10 images
for image_path in image_paths:
    # Lire l'image
    image = cv2.imread(image_path)

    # Redimensionner l'image pour la détection plus rapide (peut ne pas être nécessaire)
    height, width, channels = image.shape
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (800, 800), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Informations sur les boîtes englobantes, les classes et les confiances
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Filtre de confiance (ajustez selon vos besoins)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supprimer les détections multiples en utilisant la non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Compter le nombre de chaque élément détecté
    elements_count = {}
    for i in indexes:
        label = classes[class_ids[i]]
        elements_count[label] = elements_count.get(label, 0) + 1

    # Affichage des résultats
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 0)  # Couleur du texte (noir)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5), font, 1, color, 1)

    # Afficher la phrase "Objects detected:" suivie des aliments et de leur nombre en bas de l'image
    text = "Objects detected:   " + ", ".join([f"{label}: {elements_count[label]}" for label in elements_count])
    cv2.putText(image, text, (10, height - 20), font, 1, color, 1)

    # Afficher l'image résultante
    cv2.imshow("Image", image)
    cv2.waitKey(0)

# Fermer les fenêtres après le traitement de toutes les images
cv2.destroyAllWindows()
