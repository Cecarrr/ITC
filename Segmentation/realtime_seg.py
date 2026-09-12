from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.onnx", task="segment")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    exit()


colors = [
    (0, 255, 0),      # Hijau
    (255, 0, 0),      # Biru
    (0, 0, 255),      # Merah
    (0, 255, 255),    # Kuning
    (255, 0, 255),    # Magenta
    (255, 255, 0)     # Cyan
]

color_index = 0



def change_color(event, x, y, flags, param):

    global color_index

    if event == cv2.EVENT_LBUTTONDOWN:

        
        if 10 <= x <= 180 and 10 <= y <= 60:

            color_index += 1

            if color_index >= len(colors):
                color_index = 0



window_name = "YOLO Segmentation ONNX"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, change_color)



while True:

    ret, frame = cap.read()

    if not ret:
        print("Gagal membaca frame.")
        break


    frame = cv2.flip(frame, 1)


    results = model(
        frame,
        verbose=False,
        conf=0.711,
        retina_masks=True
    )

    result_frame = frame.copy()


    for r in results:

        if r.masks is not None:

            masks = r.masks.data.cpu().numpy()

            for mask in masks:

                
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0])
                )

                mask = mask > 0.5

                
                color = np.array(colors[color_index])

               
                alpha = 0.45

                
                overlay = np.zeros_like(frame)
                overlay[:] = color

                
                result_frame[mask] = (
                    result_frame[mask] * (1 - alpha)
                    + overlay[mask] * alpha
                ).astype(np.uint8)


    cv2.rectangle(
        result_frame,
        (10, 10),
        (180, 60),
        (50, 50, 50),
        -1
    )

    cv2.putText(
        result_frame,
        "GANTI WARNA",
        (25, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        window_name,
        result_frame
    )

  
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()