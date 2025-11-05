import cv2
import numpy as np
import random
import mediapipe as mp
import math

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Game setup ---
width, height = 640, 480
snake_size = 20
snake = [(100, 100), (80, 100), (60, 100)]
food = (random.randrange(100, width, snake_size), random.randrange(100, height, snake_size))

score = 0
direction = "RIGHT"
speed = 8  # movement pixels per frame
game_over = False

cap = cv2.VideoCapture(0)
cap.set(3, width+100)
cap.set(4, height+100)

# --- Helper functions ---
def draw_snake(frame, snake):

    for (x, y) in snake:
        cv2.rectangle(frame, (x, y), (x + snake_size, y + snake_size), (0, 255, 0), -1)

def check_collision(snake):
    head = snake[0]
    if head[0] < 0 or head[0] >= width or head[1] < 0 or head[1] >= height:
        return True
    # if head in snake[1:]:
    #     return True
    return False

def generate_food():
    return (random.randrange(0, width, snake_size), random.randrange(0, height, snake_size))

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

    # Draw the food
    cv2.rectangle(frame, food, (food[0] + snake_size, food[1] + snake_size), (0, 0, 255), -1)

    # Finger control
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (x_index, y_index), 10, (255, 255, 0), -1)

            # Control direction based on index position relative to snake head
            head_x, head_y = snake[0]
            dx, dy = x_index - head_x, y_index - head_y
            if abs(dx) > abs(dy):  # Move horizontally
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:  # Move vertically
                direction = "DOWN" if dy > 0 else "UP"

    # Move the snake
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

    # Check if the snake eats food
    if abs(snake[0][0] - food[0]) < snake_size and abs(snake[0][1] - food[1]) < snake_size:
        score += 1
        food = generate_food()
    else:
        snake.pop()

    # Check collisions
    if check_collision(snake):
        game_over = True

    # Draw snake and text
    draw_snake(frame, snake)
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        cv2.putText(frame, "GAME OVER", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Snake Finger Control", frame)
        cv2.waitKey(2000)
        snake = [(100, 100), (80, 100), (60, 100)]
        direction = "RIGHT"
        score = 0
        food = generate_food()
        game_over = False
        continue

    cv2.imshow("Snake Finger Control", frame)
    if cv2.waitKey(10) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
