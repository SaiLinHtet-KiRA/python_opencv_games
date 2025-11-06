import cv2
import numpy as np
import random
import mediapipe as mp
import tkinter as tk

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# --- Game setup ---
width, height = screen_width - 100, screen_height - 100
snake_size = 20
snake = [(100, 100), (80, 100), (60, 100)]
food = (random.randrange(0, width - snake_size, snake_size),
        random.randrange(0, height - snake_size, snake_size))

# Load and resize apple PNG (with alpha channel)
apple_size = 60
app_img = cv2.imread("images/apple.png", cv2.IMREAD_UNCHANGED)
app = cv2.resize(app_img, (apple_size, apple_size))

# --- Snake head setup (2x bigger than body) ---
snake_head_img = cv2.imread("images/snake head.png", cv2.IMREAD_UNCHANGED)
head_size = snake_size * 2  # head is twice as large
snake_head = cv2.resize(snake_head_img, (head_size, head_size))

score = 0
direction = "RIGHT"
speed = 8
game_over = False
you_win = False  # 🏆 Win flag
WIN_SCORE = 20   # 🏆 Score needed to win

cap = cv2.VideoCapture(0)
cap.set(3, width + 100)
cap.set(4, height + 100)

# --- Helper functions ---
def rotate_head(image, direction):
    """Rotate the snake head image based on direction"""
    if direction == "UP":
        angle = 0
    elif direction == "DOWN":
        angle = 180
    elif direction == "LEFT":
        angle = 90
    elif direction == "RIGHT":
        angle = 270

    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def overlay_image(background, overlay, x, y):
    """Overlay a PNG image with alpha channel onto a BGR background"""
    h, w = overlay.shape[:2]

    if x < 0 or y < 0 or y + h > background.shape[0] or x + w > background.shape[1]:
        return background  # Out of bounds

    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255.0  # Normalize alpha

    background[y:y+h, x:x+w] = (1 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background

def draw_snake_body(frame, snake):
    """Draw the snake's body (without the head)"""
    for (x, y) in snake[1:]:
        cv2.rectangle(frame, (x, y), (x + snake_size, y + snake_size), (0, 255, 0), -1)

def draw_snake_head(frame, snake, direction):
    """Draw the rotated snake head (always on top)"""
    rotated_head = rotate_head(snake_head, direction)
    head_x, head_y = snake[0]
    offset = (head_size - snake_size) // 2
    return overlay_image(frame, rotated_head, head_x - offset, head_y - offset)

def check_collision(snake):
    head = snake[0]
    if head[0] < 0 or head[0] >= width or head[1] < 0 or head[1] >= height:
        return True
    return False

def generate_food():
    return (
        random.randrange(50, width - 50 - snake_size, snake_size),
        random.randrange(25, height - 25 - snake_size, snake_size)
    )

def reset_game():
    """Reset the game after winning or losing"""
    global snake, direction, score, food, game_over, you_win
    snake = [(100, 100), (80, 100), (60, 100)]
    direction = "RIGHT"
    score = 0
    food = generate_food()
    game_over = False
    you_win = False

# --- Main loop ---
print("🖐️ Move your index finger to control the snake!")
print("ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Win Screen 🏆 ---
    if you_win:
        cv2.putText(frame, "🎉 YOU WIN! 🎉", (width // 3, height // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
        cv2.putText(frame, f"Final Score: {score}", (width // 3 + 50, height // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, "Press 'R' to Restart or 'ESC' to Exit",
                    (width // 3 - 60, height // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Snake Finger Control", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reset_game()
        continue

    # --- Game Over Screen ---
    if game_over:
        cv2.putText(frame, "GAME OVER", (width // 3, height // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        cv2.putText(frame, "Press 'R' to Restart or 'ESC' to Exit",
                    (width // 3 - 80, height // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Snake Finger Control", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reset_game()
        continue

    # --- Finger control ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (x_index, y_index), 10, (255, 255, 0), -1)

            # Change direction
            head_x, head_y = snake[0]
            dx, dy = x_index - head_x, y_index - head_y
            if abs(dx) > abs(dy):
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"

    # --- Move snake ---
    head_x, head_y = snake[0]
    if direction == "UP":
        head_y -= speed
    elif direction == "DOWN":
        head_y += speed
    elif direction == "LEFT":
        head_x -= speed
    elif direction == "RIGHT":
        head_x += speed
    new_head = (int(head_x), int(head_y))
    snake.insert(0, new_head)

    # --- Eat food ---
    if abs(snake[0][0] - food[0]) < apple_size and abs(snake[0][1] - food[1]) < apple_size:
        score += 1
        food = generate_food()
        if score >= WIN_SCORE:  # 🏆 Check win condition
            you_win = True
    else:
        snake.pop()

    # --- Check collisions ---
    if check_collision(snake):
        game_over = True

    # --- Draw everything ---
    draw_snake_body(frame, snake)
    frame = overlay_image(frame, app, food[0], food[1])
    cv2.putText(frame, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frame = draw_snake_head(frame, snake, direction)

    # --- Display ---
    cv2.imshow("Snake Finger Control", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
