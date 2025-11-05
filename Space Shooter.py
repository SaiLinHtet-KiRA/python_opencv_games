import cv2
import mediapipe as mp
import random
import time
import tkinter as tk
import numpy as np

# --- Get screen resolution ---
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# --- Game settings ---
WIDTH, HEIGHT = screen_width, screen_height
PLAYER_WIDTH, PLAYER_HEIGHT = 80, 50
BULLET_WIDTH, BULLET_HEIGHT = 5, 10
ENEMY_WIDTH, ENEMY_HEIGHT = 50, 30

BULLET_SPEED = 10
ENEMY_SPEED = 5
SPAWN_RATE = 30  # lower = more frequent
FIRE_INTERVAL = 0.5  # seconds between shots

# --- Load images ---
player_img = cv2.imread("images/space ship.png", cv2.IMREAD_UNCHANGED)
enemy_img = cv2.imread("images/space stone.png", cv2.IMREAD_UNCHANGED)

# Resize images
player_img = cv2.resize(player_img, (PLAYER_WIDTH, PLAYER_HEIGHT))
enemy_img = cv2.resize(enemy_img, (ENEMY_WIDTH, ENEMY_HEIGHT))

# --- Initialize OpenCV & MediaPipe ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def overlay_image_alpha(img, img_overlay, x, y):
    """Overlay img_overlay on top of img at position (x, y) with alpha channel."""
    h, w = img_overlay.shape[:2]

    # Clip overlay if it goes out of frame
    if x < 0:
        img_overlay = img_overlay[:, -x:]
        w += x
        x = 0
    if y < 0:
        img_overlay = img_overlay[-y:, :]
        h += y
        y = 0
    if x + w > img.shape[1]:
        img_overlay = img_overlay[:, :img.shape[1] - x]
        w = img_overlay.shape[1]
    if y + h > img.shape[0]:
        img_overlay = img_overlay[:img.shape[0] - y, :]
        h = img_overlay.shape[0]

    if h <= 0 or w <= 0:
        return

    # Apply alpha blending
    if img_overlay.shape[2] == 4:
        alpha_mask = img_overlay[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+h, x:x+w, c] = (alpha_mask * img_overlay[:, :, c] +
                                    (1 - alpha_mask) * img[y:y+h, x:x+w, c])
    else:
        img[y:y+h, x:x+w] = img_overlay


def rotate_image(image, angle):
    """Rotate an image (with alpha) while keeping size."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    return rotated


def reset_game():
    """Reset all game variables."""
    return WIDTH // 2, HEIGHT - 100, [], [], 0, time.time()


# --- Initialize game state ---
player_x, player_y, bullets, enemies, score, last_fire_time = reset_game()
game_over = False

# --- Game loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ✅ Resize webcam frame to fit full screen
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # --- Game Over Screen ---
    if game_over:
        cv2.putText(frame, f"GAME OVER! Score: {score}",
                    (WIDTH // 3, HEIGHT // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "Press 'R' to Restart or 'ESC' to Exit",
                    (WIDTH // 3 - 50, HEIGHT // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Space Shooter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            player_x, player_y, bullets, enemies, score, last_fire_time = reset_game()
            game_over = False
        continue

    # --- Hand tracking (move player) ---
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            finger_x = int(handLms.landmark[8].x * WIDTH)
            player_x = finger_x - PLAYER_WIDTH // 2
            player_x = max(0, min(WIDTH - PLAYER_WIDTH, player_x))
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # --- Fire bullets automatically ---
    current_time = time.time()
    if current_time - last_fire_time >= FIRE_INTERVAL:
        bullets.append([player_x + PLAYER_WIDTH // 2, player_y])
        last_fire_time = current_time

    # --- Update bullets ---
    bullets = [[bx, by - BULLET_SPEED] for bx, by in bullets if by > 0]

    # --- Spawn enemies with random spin ---
    if random.randint(1, SPAWN_RATE) == 1:
        ex = random.randint(0, WIDTH - ENEMY_WIDTH)
        angle = random.randint(0, 360)
        enemies.append([ex, -ENEMY_HEIGHT, angle])

    # --- Update enemies ---
    new_enemies = []
    for e in enemies:
        ex, ey, angle = e
        ey += ENEMY_SPEED
        angle = (angle + 5) % 360
        if ey < HEIGHT:
            new_enemies.append([ex, ey, angle])
    enemies = new_enemies

    # --- Collision detection ---
    for e in enemies[:]:
        ex, ey, _ = e
        for b in bullets[:]:
            bx, by = b
            if ex < bx < ex + ENEMY_WIDTH and ey < by < ey + ENEMY_HEIGHT:
                enemies.remove(e)
                bullets.remove(b)
                score += 1
                break
        # Player collision
        if (player_x < ex + ENEMY_WIDTH and player_x + PLAYER_WIDTH > ex and
                player_y < ey + ENEMY_HEIGHT and player_y + PLAYER_HEIGHT > ey):
            game_over = True

    # --- Draw player ---
    overlay_image_alpha(frame, player_img, player_x, player_y)

    # --- Draw bullets ---
    for bx, by in bullets:
        cv2.rectangle(frame, (bx, by), (bx + BULLET_WIDTH, by + BULLET_HEIGHT), (0, 255, 255), -1)

    # --- Draw spinning enemies ---
    for ex, ey, angle in enemies:
        rotated_enemy = rotate_image(enemy_img, angle)
        overlay_image_alpha(frame, rotated_enemy, ex, ey)

    # --- Draw score ---
    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # --- Display the frame ---
    cv2.imshow("Space Shooter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
