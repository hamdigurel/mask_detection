import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Yüz algılama modeli
face_net = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

# Maske tespit modeli
mask_model = load_model("mask_detector.model")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104, 177, 123)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = face.reshape(1, 224, 224, 3)

            mask, no_mask = mask_model.predict(face)[0]

            label = "Maskeli" if mask > no_mask else "Maskesiz"
            color = (0, 255, 0) if label == "Maskeli" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Maske Tespiti", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
