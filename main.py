import cv2
import mediapipe as mp
import sys
import math
import time
import pygame
from collections import deque

from pygame.examples import sound

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
pygame.mixer.init()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara. Verifica los permisos o el índice.")
    sys.exit()


def distance(base, target):
    dx = base.x - target.x
    dy = base.y - target.y
    dz = base.z - target.z
    return math.sqrt(dx**2 + dy**2 + dz**2)

def angle_finger(finger_0, finger_1, finger_4):
    a = distance(finger_0, finger_4)
    b = distance(finger_0, finger_1)
    c = distance(finger_4, finger_1)
    angle = (a**2 - b**2 - c**2) / (-2 * b * c)
    angle = max(-1.0, min(1.0, angle))
    return math.degrees(math.acos(angle))

def closed_finger(finger_0, finger_1, finger_4):
    deg = angle_finger(finger_0, finger_1, finger_4)
    return not (0 < deg < 55)


def classify_note(nodes, image_shape):
    h, w = image_shape[:2]

    wrist  = nodes[0]

    # Pulgar
    thumb_2 = nodes[2]
    thumb_3 = nodes[3]
    thumb_4 = nodes[4]
    # Índice
    index_0 = nodes[5]
    index_1 = nodes[6]
    index_3 = nodes[8]

    # Medio
    middle_0 = nodes[9]
    middle_1 = nodes[10]
    middle_3 = nodes[12]

    # Anular
    ring_0 = nodes[13]
    ring_1 = nodes[14]
    ring_3 = nodes[16]

    # Meñique
    little_0 = nodes[17]
    little_1 = nodes[18]
    little_3 = nodes[20]

    # Pulgar: tip
    thumb_tip = nodes[4]
    thumb_mcp = nodes[2]

    # Ángulos
    angle_index  = angle_finger(wrist, index_1,  index_3)
    angle_middle = angle_finger(wrist, middle_1, middle_3)
    angle_ring   = angle_finger(wrist, ring_1,   ring_3)
    angle_little = angle_finger(wrist, little_1, little_3)

    index_open  = closed_finger(wrist, index_1,  index_3)
    middle_open = closed_finger(wrist, middle_1, middle_3)
    ring_open   = closed_finger(wrist, ring_1,   ring_3)
    little_open = closed_finger(wrist, little_1, little_3)

    thumb_up = (thumb_tip.y < thumb_mcp.y - 0.04)

    if (thumb_tip.y < index_3.y < middle_3.y < ring_3.y < little_3.y and index_open and middle_open and ring_open and little_open):
        return "sol"

    if (angle_index >= 160 and angle_middle >= 160
            and angle_ring >= 170 and angle_little >= 170):
        return "re"

    if (110 < angle_index  < 160 and 110 < angle_middle < 160
            and 110 < angle_ring   < 170 and 110 < angle_little < 170):
        return "mi"

    if (not thumb_up and not index_open and not middle_open
            and not ring_open and not little_open):
        return "fa"


    if (thumb_up and not index_open and not middle_open
            and not ring_open and not little_open):
        return "do"


    if (35 < angle_index  < 130 and 35 < angle_middle < 130
            and 35 < angle_ring   < 140 and 35 < angle_little < 140):
        return "la"


    if (index_open and not middle_open
            and not ring_open and not little_open
            and angle_index > 110 and angle_middle < 20):
        return "si"



    return None


NOTE_DATA = {
    "do":  {"color": (219, 84,  97)},
    "re":  {"color": (255, 160, 50)},
    "mi":  {"color": (255, 220, 50)},
    "fa":  {"color": (80,  200, 120)},
    "sol": {"color": (50,  180, 255)},
    "la":  {"color": (160, 100, 255)},
    "si":  {"color": (255, 120, 200)},
}
NOTE_ORDER = ["do", "re", "mi", "fa", "sol", "la", "si"]

# ─── Estabilizador de nota (evita parpadeo) ───────────────────────────────────

class NoteStabilizer:
    def __init__(self, window=8, threshold=5):
        self.history = deque(maxlen=window)
        self.threshold = threshold
        self.stable = None
        self.confidence = 0.0

    def update(self, note):
        self.history.append(note)
        if len(self.history) < self.threshold:
            return self.stable
        counts = {}
        for n in self.history:
            if n:
                counts[n] = counts.get(n, 0) + 1
        if not counts:
            self.stable = None
            self.confidence = 0.0
            return None
        best, cnt = max(counts.items(), key=lambda x: x[1])
        self.confidence = cnt / len(self.history)
        if cnt >= self.threshold:
            self.stable = best
        return self.stable

# ─── UI ───────────────────────────────────────────────────────────────────────

PANEL_W = 270

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    """Rectángulo redondeado con alpha."""
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        img[:] = overlay

def draw_ui(frame, stable_note, confidence, fps):
    h, w = frame.shape[:2]
    px = w - PANEL_W  # inicio del panel lateral

    # ── Fondo del panel ───────────────────────────────────────────────────────
    panel_overlay = frame.copy()
    cv2.rectangle(panel_overlay, (px, 0), (w, h), (12, 14, 22), -1)
    cv2.addWeighted(panel_overlay, 0.82, frame, 0.18, 0, frame)

    # Línea borde izquierda del panel
    cv2.line(frame, (px, 0), (px, h), (40, 44, 60), 2)

    # ── Título ────────────────────────────────────────────────────────────────
    cv2.putText(frame, "SOLFEO", (px + 18, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 220, 255), 1)
    cv2.putText(frame, "Kodaly", (px + 18, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 160), 1)
    cv2.line(frame, (px + 10, 70), (w - 10, 70), (40, 44, 65), 1)

    # ── Lista de notas ────────────────────────────────────────────────────────
    for i, note_name in enumerate(NOTE_ORDER):
        data = NOTE_DATA[note_name]
        color = data["color"]
        is_active = (stable_note == note_name)
        y = 90 + i * 58

        if is_active:
            # Fondo resaltado
            ov = frame.copy()
            cv2.rectangle(ov, (px + 6, y - 16), (w - 6, y + 34),
                          tuple(int(c * 0.22) for c in color), -1)
            cv2.addWeighted(ov, 0.9, frame, 0.1, 0, frame)
            cv2.rectangle(frame, (px + 6, y - 16), (w - 6, y + 34), color, 2)

        # Bolita de color
        dot_c = color if is_active else (45, 48, 65)
        cv2.circle(frame, (px + 22, y + 8), 7, dot_c, -1)
        if is_active:
            cv2.circle(frame, (px + 22, y + 8), 10, color, 1)

        # Nombre de la nota
        tc = color if is_active else (110, 115, 150)
        fw = 2 if is_active else 1
        cv2.putText(frame, note_name, (px + 36, y + 13),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, tc, fw)

    cv2.line(frame, (px + 10, 90 + 7*58), (w - 10, 90 + 7*58), (40, 44, 65), 1)

    # ── Nota detectada grande (abajo izquierda) ───────────────────────────────
    box_y1, box_y2 = h - 130, h - 10
    if stable_note and stable_note in NOTE_DATA:
        data  = NOTE_DATA[stable_note]
        color = data["color"]

        # Fondo
        bg_ov = frame.copy()
        cv2.rectangle(bg_ov, (10, box_y1), (px - 10, box_y2),
                      tuple(int(c * 0.18) for c in color), -1)
        cv2.addWeighted(bg_ov, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, box_y1), (px - 10, box_y2), color, 2)

        # Nombre enorme
        cv2.putText(frame, stable_note, (30, box_y2 - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 3.2, color, 4)


        # Barra de confianza
        bar_max = px - 70
        bar_fill = int(bar_max * confidence)
        cv2.rectangle(frame, (30, box_y1 + 8), (30 + bar_max, box_y1 + 18),
                      (30, 33, 50), -1)
        cv2.rectangle(frame, (30, box_y1 + 8), (30 + bar_fill, box_y1 + 18),
                      color, -1)
        pct_txt = f"{int(confidence*100)}%"
        cv2.putText(frame, pct_txt, (30 + bar_max + 6, box_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    else:
        cv2.rectangle(frame, (10, box_y1), (px - 10, box_y2), (35, 38, 55), 1)
        cv2.putText(frame, "---", (30, box_y2 - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (60, 65, 90), 2)
        cv2.putText(frame, "Muestra un gesto", (30, box_y2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 75, 100), 1)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS {fps:.0f}", (px + 18, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 60, 85), 1)

    # ── Instrucción salida ────────────────────────────────────────────────────
    cv2.putText(frame, "Q / ESC : salir", (14, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 75, 100), 1)

    return frame


stabilizer = NoteStabilizer(window=8, threshold=5)
prev_time = time.time()

# Estilo de landmarks personalizado
lm_style = mp_drawing.DrawingSpec(color=(0, 220, 180), thickness=2, circle_radius=3)
conn_style = mp_drawing.DrawingSpec(color=(0, 120, 100), thickness=2)

note = ""

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando frame vacío de la cámara.")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image.flags.writeable = True

        raw_note = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    lm_style, conn_style)

                nodes = hand_landmarks.landmark
                raw_note = classify_note(nodes, image.shape)


            if raw_note:
                if raw_note != note:
                    note = raw_note
                    sound = pygame.mixer.Sound(str(raw_note) + ".wav")
                    sound.set_volume(0.5)
                    sound.play()

        stable = stabilizer.update(raw_note)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        image = draw_ui(image, stable, stabilizer.confidence, fps)
        cv2.imshow("Solfeo Kodaly - Reconocimiento de Notas", image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

cap.release()
cv2.destroyAllWindows()