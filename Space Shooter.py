import cv2
import mediapipe as mp
import random
import time

# --- Game settings ---
WIDTH, HEIGHT = 640, 480
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 30
BULLET_WIDTH, BULLET_HEIGHT = 5, 10
ENEMY_WIDTH, ENEMY_HEIGHT = 50, 30

PLAYER_COLOR = (0, 255, 0)
BULLET_COLOR = (0, 255, 255)
ENEMY_COLOR = (0, 0, 255)
BG_COLOR = (0, 0, 0)

BULLET_SPEED = 10
ENEMY_SPEED = 5
SPAWN_RATE = 30  # lower = more frequent

# --- Initialize OpenCV & MediaPipe ---
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def reset_game():
    """Reset all game variables."""
    return WIDTH // 2, HEIGHT - 50, [], [], 0, time.time()

# --- Initialize game state ---
player_x, player_y, bullets, enemies, score, last_fire_time = reset_game()
fire_interval = 0.5  # seconds

game_over = False

# --- Game loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # --- If game over ---
    if game_over:
        cv2.putText(frame, f"GAME OVER! Score: {score}", (100, HEIGHT // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "Press 'R' to Restart or 'ESC' to Exit",
                    (60, HEIGHT // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Space Shooter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('r'):
            player_x, player_y, bullets, enemies, score, last_fire_time = reset_game()
            game_over = False
        continue

    # --- Finger detection ---
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = frame.shape
            finger_x = int(handLms.landmark[8].x * w)  # Index fingertip
            player_x = finger_x - PLAYER_WIDTH // 2
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # --- Bound player inside screen ---
    player_x = max(0, min(WIDTH - PLAYER_WIDTH, player_x))

    # --- Fire bullets every interval ---
    current_time = time.time()
    if current_time - last_fire_time >= fire_interval:
        bullets.append([player_x + PLAYER_WIDTH // 2, player_y])
        last_fire_time = current_time

    # --- Update bullets ---
    new_bullets = []
    for b in bullets:
        b[1] -= BULLET_SPEED
        if b[1] > 0:
            new_bullets.append(b)
    bullets = new_bullets

    # --- Spawn enemies ---
    if random.randint(1, SPAWN_RATE) == 1:
        ex = random.randint(0, WIDTH - ENEMY_WIDTH)
        enemies.append([ex, -ENEMY_HEIGHT])

    # --- Update enemies ---
    new_enemies = []
    for e in enemies:
        e[1] += ENEMY_SPEED
        if e[1] < HEIGHT:
            new_enemies.append(e)
    enemies = new_enemies

    # --- Collision detection ---
    for e in enemies[:]:
        ex, ey = e
        # Check bullets
        for b in bullets[:]:
            bx, by = b
            if ex < bx < ex + ENEMY_WIDTH and ey < by < ey + ENEMY_HEIGHT:
                enemies.remove(e)
                bullets.remove(b)
                score += 1
                break
        # Check player collision
        if (player_x < ex + ENEMY_WIDTH and player_x + PLAYER_WIDTH > ex and
                player_y < ey + ENEMY_HEIGHT and player_y + PLAYER_HEIGHT > ey):
            game_over = True

    # --- Draw player ---
    cv2.rectangle(frame, (player_x, player_y),
                  (player_x + PLAYER_WIDTH, player_y + PLAYER_HEIGHT), PLAYER_COLOR, -1)

    # --- Draw bullets ---
    for b in bullets:
        cv2.rectangle(frame, (b[0], b[1]), (b[0] + BULLET_WIDTH, b[1] + BULLET_HEIGHT), BULLET_COLOR, -1)

    # --- Draw enemies ---
    for e in enemies:
        cv2.rectangle(frame, (e[0], e[1]), (e[0] + ENEMY_WIDTH, e[1] + ENEMY_HEIGHT), ENEMY_COLOR, -1)

    # --- Draw score ---
    cv2.putText(frame, f"Score: {score}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Space Shooter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
