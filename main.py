import cv2
import mediapipe as mp
import sys
import math
import time
import pygame
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pygame.mixer.init()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    sys.exit()

NOTE_ORDER = ["do", "re", "mi", "fa", "sol", "la", "si"]


def get_chord_notes(root_note, num_fingers):

    if root_note not in NOTE_ORDER:
        return [root_note] if root_note else []
    idx = NOTE_ORDER.index(root_note)
    if num_fingers == 1:
        return [NOTE_ORDER[idx]]
    elif num_fingers == 2:
        return [NOTE_ORDER[idx], NOTE_ORDER[(idx + 2) % 7]]
    elif num_fingers == 3:
        return [NOTE_ORDER[idx], NOTE_ORDER[(idx + 2) % 7], NOTE_ORDER[(idx + 4) % 7]]
    elif num_fingers >= 4:
        return [NOTE_ORDER[idx], NOTE_ORDER[(idx + 2) % 7],
                NOTE_ORDER[(idx + 4) % 7], NOTE_ORDER[(idx + 6) % 7]]
    return [root_note]

CHORD_NAMES = {
    ("do",  3): "C",      ("do",  4): "Cmaj7",
    ("re",  3): "Dm",     ("re",  4): "Dm7",
    ("mi",  3): "Em",     ("mi",  4): "Em7",
    ("fa",  3): "F",      ("fa",  4): "Fmaj7",
    ("sol", 3): "G",      ("sol", 4): "G7",
    ("la",  3): "Am",     ("la",  4): "Am7",
    ("si",  3): "Bdim",   ("si",  4): "Bm7b5",
}


def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def angle_finger(base, mid, tip):
    a = distance(base, tip)
    b = distance(base, mid)
    c = distance(tip, mid)
    val = (a**2 - b**2 - c**2) / (-2 * b * c)
    val = max(-1.0, min(1.0, val))
    return math.degrees(math.acos(val))

def is_finger_open(wrist, pip, tip):
    deg = angle_finger(wrist, pip, tip)
    return 0 < deg < 55  # cerrado si NOT en este rango → abierto si SÍ

def finger_open(wrist, pip, tip):
    deg = angle_finger(wrist, pip, tip)
    return not (0 < deg < 55)  # True = abierto (igual que el original)


def classify_note(nodes):
    wrist    = nodes[0]
    thumb_tip = nodes[4]; thumb_mcp = nodes[2]
    index_1  = nodes[6];  index_3  = nodes[8]
    middle_1 = nodes[10]; middle_3 = nodes[12]
    ring_1   = nodes[14]; ring_3   = nodes[16]
    little_1 = nodes[18]; little_3 = nodes[20]

    angle_index  = angle_finger(wrist, index_1,  index_3)
    angle_middle = angle_finger(wrist, middle_1, middle_3)
    angle_ring   = angle_finger(wrist, ring_1,   ring_3)
    angle_little = angle_finger(wrist, little_1, little_3)

    index_open  = finger_open(wrist, index_1,  index_3)
    middle_open = finger_open(wrist, middle_1, middle_3)
    ring_open   = finger_open(wrist, ring_1,   ring_3)
    little_open = finger_open(wrist, little_1, little_3)
    thumb_up    = thumb_tip.y < thumb_mcp.y - 0.04

    if (thumb_tip.y < index_3.y < middle_3.y < ring_3.y < little_3.y
            and index_open and middle_open and ring_open and little_open):
        return "sol"
    if angle_index >= 160 and angle_middle >= 160 and angle_ring >= 170 and angle_little >= 170:
        return "re"
    if (110 < angle_index < 160 and 110 < angle_middle < 160
            and 110 < angle_ring < 170 and 110 < angle_little < 170):
        return "mi"
    if not thumb_up and not index_open and not middle_open and not ring_open and not little_open:
        return "fa"
    if thumb_up and not index_open and not middle_open and not ring_open and not little_open:
        return "do"
    if (35 < angle_index < 130 and 35 < angle_middle < 130
            and 35 < angle_ring < 140 and 35 < angle_little < 140):
        return "la"
    if (index_open and not middle_open and not ring_open and not little_open
            and angle_index > 110 and angle_middle < 20):
        return "si"
    return None


def count_open_fingers_left(nodes):

    wrist    = nodes[0]
    pairs = [
        (nodes[6],  nodes[8]),   # índice
        (nodes[10], nodes[12]),  # medio
        (nodes[14], nodes[16]),  # anular
        (nodes[18], nodes[20]),  # meñique
    ]
    count = 0
    for pip, tip in pairs:
        if finger_open(wrist, pip, tip):
            count += 1
    return max(1, count)  # mínimo 1 para que siempre suene algo


def wrist_to_volume(wrist_y):

    vol = 1.0 - wrist_y  # invertir: arriba = 1.0, abajo = 0.0
    return max(0.05, min(1.0, vol))


def load_sound(name):
    try:
        return pygame.mixer.Sound(f"{name}.wav")
    except Exception as e:
        print(f"Advertencia: no se pudo cargar {name}.wav — {e}")
        return None

SOUNDS = {note: load_sound(note) for note in NOTE_ORDER}

def play_chord(notes, volume):
    for n in notes:
        s = SOUNDS.get(n)
        if s:
            s.set_volume(volume)
            s.play()


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
            self.stable = None; self.confidence = 0.0; return None
        best, cnt = max(counts.items(), key=lambda x: x[1])
        self.confidence = cnt / len(self.history)
        if cnt >= self.threshold:
            self.stable = best
        return self.stable


PANEL_W = 290
NOTE_DATA = {
    "do":  {"color": (219, 84,  97)},
    "re":  {"color": (255, 160, 50)},
    "mi":  {"color": (255, 220, 50)},
    "fa":  {"color": (80,  200, 120)},
    "sol": {"color": (50,  180, 255)},
    "la":  {"color": (160, 100, 255)},
    "si":  {"color": (255, 120, 200)},
}

def draw_volume_bar(frame, volume, x, y, w, h_bar):
    cv2.rectangle(frame, (x, y), (x + w, y + h_bar), (30, 33, 50), -1)
    fill = int(h_bar * volume)
    color = (50, 220, 180)
    cv2.rectangle(frame, (x, y + h_bar - fill), (x + w, y + h_bar), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h_bar), (60, 65, 85), 1)
    cv2.putText(frame, "VOL", (x - 2, y + h_bar + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 90, 110), 1)

def draw_ui(frame, stable_note, confidence, fps, num_fingers, volume, chord_notes, chord_name):
    h, w = frame.shape[:2]
    px = w - PANEL_W

    # Panel lateral
    ov = frame.copy()
    cv2.rectangle(ov, (px, 0), (w, h), (12, 14, 22), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (px, 0), (px, h), (40, 44, 60), 2)

    # Título
    cv2.putText(frame, "SOLFEO", (px + 18, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 220, 255), 1)
    cv2.putText(frame, "Kodaly  |  C Mayor", (px + 18, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 160), 1)
    cv2.line(frame, (px + 10, 70), (w - 10, 70), (40, 44, 65), 1)

    # Lista de notas
    for i, note_name in enumerate(NOTE_ORDER):
        color = NOTE_DATA[note_name]["color"]
        is_root = (stable_note == note_name)
        in_chord = note_name in chord_notes
        y = 90 + i * 52

        if is_root:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (px + 6, y - 14), (w - 6, y + 30),
                          tuple(int(c * 0.22) for c in color), -1)
            cv2.addWeighted(ov2, 0.9, frame, 0.1, 0, frame)
            cv2.rectangle(frame, (px + 6, y - 14), (w - 6, y + 30), color, 2)
        elif in_chord:
            # Notas del acorde pero no la raíz → resaltado suave
            ov2 = frame.copy()
            cv2.rectangle(ov2, (px + 6, y - 14), (w - 6, y + 30),
                          tuple(int(c * 0.10) for c in color), -1)
            cv2.addWeighted(ov2, 0.7, frame, 0.3, 0, frame)

        dot_c = color if (is_root or in_chord) else (45, 48, 65)
        cv2.circle(frame, (px + 22, y + 8), 7, dot_c, -1)
        if is_root:
            cv2.circle(frame, (px + 22, y + 8), 10, color, 1)

        tc = color if is_root else ((200, 200, 220) if in_chord else (110, 115, 150))
        fw = 2 if is_root else 1
        cv2.putText(frame, note_name, (px + 36, y + 13),
                    cv2.FONT_HERSHEY_DUPLEX, 0.68, tc, fw)

    cv2.line(frame, (px + 10, 90 + 7*52 + 4), (w - 10, 90 + 7*52 + 4), (40, 44, 65), 1)

    fy = 90 + 7*52 + 18
    cv2.putText(frame, "Mano izq:", (px + 10, fy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)
    for di in range(4):
        bx = px + 10 + di * 34
        bc = (80, 200, 120) if (di < num_fingers) else (35, 38, 55)
        cv2.circle(frame, (bx + 10, fy + 34), 12, bc, -1)
        cv2.putText(frame, str(di + 1), (bx + 6, fy + 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0) if (di < num_fingers) else (50, 55, 75), 1)

    if chord_name:
        cv2.putText(frame, chord_name, (px + 155, fy + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (220, 220, 100), 2)

    # ── Zona inferior: nota grande + barra de volumen ─────────────────────────
    box_y1, box_y2 = h - 130, h - 10
    if stable_note and stable_note in NOTE_DATA:
        color = NOTE_DATA[stable_note]["color"]
        bg_ov = frame.copy()
        cv2.rectangle(bg_ov, (10, box_y1), (px - 50, box_y2),
                      tuple(int(c * 0.18) for c in color), -1)
        cv2.addWeighted(bg_ov, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (10, box_y1), (px - 50, box_y2), color, 2)
        cv2.putText(frame, stable_note, (30, box_y2 - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 3.2, color, 4)
        bar_max = px - 90
        bar_fill = int(bar_max * confidence)
        cv2.rectangle(frame, (30, box_y1 + 8), (30 + bar_max, box_y1 + 18), (30, 33, 50), -1)
        cv2.rectangle(frame, (30, box_y1 + 8), (30 + bar_fill, box_y1 + 18), color, -1)
        cv2.putText(frame, f"{int(confidence*100)}%", (30 + bar_max + 6, box_y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        if len(chord_notes) > 1:
            chord_str = " + ".join(chord_notes)
            cv2.putText(frame, chord_str, (30, box_y2 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    else:
        cv2.rectangle(frame, (10, box_y1), (px - 50, box_y2), (35, 38, 55), 1)
        cv2.putText(frame, "---", (30, box_y2 - 45),
                    cv2.FONT_HERSHEY_DUPLEX, 2.5, (60, 65, 90), 2)
        cv2.putText(frame, "Muestra gesto mano derecha", (30, box_y2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 75, 100), 1)

    # Barra de volumen vertical (a la derecha de la caja de nota)
    draw_volume_bar(frame, volume, px - 42, box_y1, 22, box_y2 - box_y1)

    # FPS + instrucciones
    cv2.putText(frame, f"FPS {fps:.0f}", (px + 18, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (55, 60, 85), 1)
    cv2.putText(frame, "Q / ESC : salir", (14, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 75, 100), 1)
    cv2.putText(frame, "Der: nota  |  Izq: dedos = modo armonico", (14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 75, 100), 1)

    return frame


stabilizer = NoteStabilizer(window=8, threshold=5)
prev_time  = time.time()

lm_style   = mp_drawing.DrawingSpec(color=(0, 220, 180), thickness=2, circle_radius=3)
conn_style = mp_drawing.DrawingSpec(color=(0, 120, 100), thickness=2)
lm_style_l = mp_drawing.DrawingSpec(color=(255, 160, 50), thickness=2, circle_radius=3)
conn_l     = mp_drawing.DrawingSpec(color=(180, 100, 20), thickness=2)

current_note      = ""
current_volume    = 0.5
num_fingers_left  = 1
current_chord_notes = []
current_chord_name  = ""

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,               # ← ahora detectamos 2 manos
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image     = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results   = hands.process(image_rgb)
        image.flags.writeable = True

        raw_note = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness):

                label = handedness.classification[0].label

                nodes = hand_landmarks.landmark

                if label == "Right":
                    mp_drawing.draw_landmarks(image, hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              lm_style, conn_style)
                    raw_note = classify_note(nodes)
                    current_volume = wrist_to_volume(nodes[0].y)

                elif label == "Left":
                    mp_drawing.draw_landmarks(image, hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              lm_style_l, conn_l)
                    num_fingers_left = count_open_fingers_left(nodes)

        stable = stabilizer.update(raw_note)

        if stable:
            current_chord_notes = get_chord_notes(stable, num_fingers_left)
            current_chord_name  = CHORD_NAMES.get((stable, num_fingers_left), stable)
            if num_fingers_left == 1:
                current_chord_name = stable  # solo nota
        else:
            current_chord_notes = []
            current_chord_name  = ""

        if stable and (stable != current_note or
                       len(current_chord_notes) != len(get_chord_notes(current_note, num_fingers_left))):
            current_note = stable
            play_chord(current_chord_notes, current_volume)
        elif stable:
            for n in current_chord_notes:
                s = SOUNDS.get(n)
                if s:
                    s.set_volume(current_volume)

        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        image = draw_ui(image, stable, stabilizer.confidence, fps,
                        num_fingers_left, current_volume,
                        current_chord_notes, current_chord_name)

        cv2.imshow("Solfeo Kodaly - Reconocimiento de Notas", image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

cap.release()
cv2.destroyAllWindows()