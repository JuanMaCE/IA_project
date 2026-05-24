import cv2
import mediapipe as mp
import sys
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 1. Configuramos la captura de video
# Nota para Linux: Si el índice 0 no funciona, prueba con 1, 2, o '/dev/video0'
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara. Verifica los permisos o el índice.")
    sys.exit()



def distance(base, target):
    distance_x = base.x - target.x
    distance_y = base.y - target.y
    distance_z = base.z - target.z
    return math.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)


def closed_finger(wrist, knuckles, fingernail):
    a = distance(wrist, fingernail)
    b = distance(wrist, knuckles)
    c = distance(fingernail, knuckles)
    angle = (a ** 2 - b ** 2 - c ** 2) / (-2 * b * c)
    angle = max(-1.0, min(1.0, angle))
    angle = math.acos(angle)
    return math.degrees(angle)




# 2. Inicializamos MediaPipe Hands en modo video
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frame vacío de la cámara.")
            continue

        # Espejamos la imagen (efecto espejo) y convertimos a RGB
        image = cv2.flip(image, 1)
        # MediaPipe requiere RGB, pero OpenCV usa BGR por defecto
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesamos el frame para buscar manos
        # Mejoramos el rendimiento marcando la imagen como no escribible antes de procesar
        image.flags.writeable = False
        results = hands.process(image_rgb)
        image.flags.writeable = True

        # Dibujamos los resultados
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                nodes = hand_landmarks.landmark

            # muñeca
            wrist = nodes[0]

            # indice
            index_knuckles = nodes[5]
            index_fingernail = nodes[8]

            # middle

            distance_x = index_knuckles.x - index_fingernail.x
            distance_y = index_knuckles.y - index_fingernail.y
            distance_z = index_knuckles.z - index_fingernail.z
            distance_finger = math.sqrt(distance_x ** 2 + distance_y ** 2 + distance_z ** 2)
            print(distance_finger)



        # Mostramos la ventana
        cv2.imshow('MediaPipe Hands - Linux', image)

        # Salir con la tecla 'q' (1 ms de espera para un flujo de video fluido)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()