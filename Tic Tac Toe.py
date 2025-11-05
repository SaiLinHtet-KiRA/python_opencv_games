import mediapipe as mp
import numpy as np
import math
import cv2

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Game setup ---
cell_size = 150
board = np.zeros((3, 3), dtype=str)
current_player = 'X'
game_over = False
tap_cooldown = 0  # prevents multiple taps from being counted at once

def draw_grid(frame):
    for i in range(1, 3):
        cv2.line(frame, (0, i * cell_size), (cell_size * 3, i * cell_size), (0, 255, 0), 2)
        cv2.line(frame, (i * cell_size, 0), (i * cell_size, cell_size * 3), (0, 255, 0), 2)

def draw_marks(frame):
    for r in range(3):
        for c in range(3):
            x_center = c * cell_size + cell_size // 2
            y_center = r * cell_size + cell_size // 2
            if board[r, c] == 'X':
                cv2.line(frame, (x_center - 40, y_center - 40), (x_center + 40, y_center + 40), (255, 0, 0), 3)
                cv2.line(frame, (x_center - 40, y_center + 40), (x_center + 40, y_center - 40), (255, 0, 0), 3)
            elif board[r, c] == 'O':
                cv2.circle(frame, (x_center, y_center), 45, (0, 0, 255), 3)

def check_winner():
    for i in range(3):
        if board[i, 0] != '' and np.all(board[i, :] == board[i, 0]):
            return board[i, 0]
        if board[0, i] != '' and np.all(board[:, i] == board[0, i]):
            return board[0, i]
    if board[0, 0] != '' and np.all(np.diag(board) == board[0, 0]):
        return board[0, 0]
    if board[0, 2] != '' and np.all(np.diag(np.fliplr(board)) == board[0, 2]):
        return board[0, 2]
    if np.all(board != ''):
        return 'Draw'
    return None

# --- Camera loop ---
cap = cv2.VideoCapture(0)
print("👋 Tap your thumb and index finger together to place your move!")
print("Press 'r' to reset or ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (cell_size * 3, cell_size * 3))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Thumb tip and index tip landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)

            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 255), -1)

            # Distance between thumb and index tip
            dist = math.hypot(x_index - x_thumb, y_index - y_thumb)

            # Detect tap gesture (threshold ~40 px)
            if dist < 40 and tap_cooldown == 0 and not game_over:
                row, col = y_index // cell_size, x_index // cell_size
                if row < 3 and col < 3 and board[row, col] == '':
                    board[row, col] = current_player
                    winner = check_winner()
                    if winner:
                        game_over = True
                        print("🎉", "Draw!" if winner == 'Draw' else f"Player {winner} wins!")
                    current_player = 'O' if current_player == 'X' else 'X'
                    tap_cooldown = 10  # add cooldown frames

    if tap_cooldown > 0:
        tap_cooldown -= 1

    draw_grid(frame)
    draw_marks(frame)

    if game_over:
        text = f"{'Draw!' if check_winner() == 'Draw' else f'Player {check_winner()} wins!'}"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Tic Tac Toe - Finger Tap", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        board[:] = ''
        current_player = 'X'
        game_over = False
        print("🔁 Game reset!")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
