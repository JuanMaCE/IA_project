import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 1. Configuramos la captura de video (0 suele ser la webcam integrada)
cap = cv2.VideoCapture(0)

# 2. Inicializamos MediaPipe Hands en modo video
with mp_hands.Hands(
        static_image_mode=False,  # Cambiado a False para optimizar el rastreo en video
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Espejamos la imagen y convertimos a RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesamos el frame
        results = hands.process(image_rgb)

        # Dibujamos los resultados
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Aquí podrías llamar a tus funciones: dedo_levantado o distancia

        # Mostramos la ventana
        cv2.imshow('MediaPipe Hands - Video', image)

        # Salir con la tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()