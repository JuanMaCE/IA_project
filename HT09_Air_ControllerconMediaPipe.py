import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import mediapipe as mp
import math
import pulsectl

#python /home/juan/PycharmProjects/IA_project/HT09_Air_ControllerconMediaPipe.py

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pulse = pulsectl.Pulse('mi-app')
sink = pulse.sink_list()[0]

angulo_derecho = 2 * math.pi / 7
angulo_izquierdo = 2 * math.pi / 8

print(angulo_izquierdo, " este es el  angulo jiji")


def generar_lineas(count_lines, image, radius_x, radius_y, radius):
        angulo_base = 2 * math.pi / count_lines
        angulo_var = angulo_base
        for i in range(count_lines):
            posicion_x = int(math.sin(angulo_var) * radius + radius_x)
            posicion_y = int(math.cos(angulo_var) * radius + radius_y)
            cv2.line(image,
                     (radius_x, radius_y),
                     (posicion_x, posicion_y),
                     (0, 255, 0), 2)
            angulo_var += angulo_base




with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        radius = 180
        # aqui genero los circulos ahora generar
        chords = cv2.circle(image, (250, 500), radius, (0, 0, 255), 3)
        type = cv2.circle(image, (1000, 500), radius, (0, 255, 0), 3)
        generar_lineas(7, image, 250, 500, radius)

        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # pulgar
                thumb = hand_landmarks.landmark[4]
                x1 = int(thumb.x * w)
                y1 = int(thumb.y * h)

                # indice
                index = hand_landmarks.landmark[8]
                x2 = int(index.x * w)
                y2 = int(index.y * h)


        # ventan
        cv2.imshow('MediaPipe Hands - Video', image)

        # q
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()