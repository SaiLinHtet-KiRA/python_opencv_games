import mediapipe as mp
import numpy as np
import math
import cv2
import tkinter as tk

# --- Get screen resolution ---
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# --- Game setup ---
cell_size = 200
grid_size = cell_size * 3  # 600x600
board = np.full((3, 3), '', dtype=str)
current_player = 'X'
game_over = False
tap_cooldown = 0

# Center of the screen
offset_x = (screen_width - grid_size) // 2
offset_y = (screen_height - grid_size) // 2

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def draw_grid(frame):
    """Draw the centered Tic Tac Toe grid."""
    for i in range(1, 3):
        # Horizontal lines
        cv2.line(frame,
                 (offset_x, offset_y + i * cell_size),
                 (offset_x + grid_size, offset_y + i * cell_size),
                 (0, 255, 0), 3)
        # Vertical lines
        cv2.line(frame,
                 (offset_x + i * cell_size, offset_y),
                 (offset_x + i * cell_size, offset_y + grid_size),
                 (0, 255, 0), 3)


def draw_marks(frame):
    """Draw X and O marks on the centered board."""
    for r in range(3):
        for c in range(3):
            x_center = offset_x + c * cell_size + cell_size // 2
            y_center = offset_y + r * cell_size + cell_size // 2
            if board[r, c] == 'X':
                cv2.line(frame, (x_center - 50, y_center - 50),
                         (x_center + 50, y_center + 50), (255, 0, 0), 5)
                cv2.line(frame, (x_center - 50, y_center + 50),
                         (x_center + 50, y_center - 50), (255, 0, 0), 5)
            elif board[r, c] == 'O':
                cv2.circle(frame, (x_center, y_center), 60, (0, 0, 255), 5)


def check_winner():
    """Check winner or draw."""
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


# --- Camera setup ---
cap = cv2.VideoCapture(0)
print("👋 Tap thumb and index finger together to place your move.")
print("Press 'r' to reset or ESC to quit.")

# Optional: Fullscreen window
cv2.namedWindow("Tic Tac Toe - Centered", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Tic Tac Toe - Centered", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror and resize webcam to fill screen
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Hand tracking ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            x_index = int(index_tip.x * screen_width)
            y_index = int(index_tip.y * screen_height)
            x_thumb = int(thumb_tip.x * screen_width)
            y_thumb = int(thumb_tip.y * screen_height)

            cv2.circle(frame, (x_index, y_index), 10, (0, 255, 255), -1)

            # Distance between thumb and index
            dist = math.hypot(x_index - x_thumb, y_index - y_thumb)

            # Tap gesture
            if dist < 40 and tap_cooldown == 0 and not game_over:
                # Check if inside grid
                if offset_x <= x_index < offset_x + grid_size and offset_y <= y_index < offset_y + grid_size:
                    col = (x_index - offset_x) // cell_size
                    row = (y_index - offset_y) // cell_size
                    if board[row, col] == '':
                        board[row, col] = current_player
                        winner = check_winner()
                        if winner:
                            game_over = True
                            print("🎉", "Draw!" if winner == 'Draw' else f"Player {winner} wins!")
                        current_player = 'O' if current_player == 'X' else 'X'
                        tap_cooldown = 10

    if tap_cooldown > 0:
        tap_cooldown -= 1

    # --- Draw game ---
    draw_grid(frame)
    draw_marks(frame)

    if game_over:
        winner = check_winner()
        text = "Draw!" if winner == 'Draw' else f"Player {winner} wins!"
        cv2.putText(frame, text, (offset_x, offset_y - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    # --- Show frame ---
    cv2.imshow("Tic Tac Toe - Centered", frame)

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
