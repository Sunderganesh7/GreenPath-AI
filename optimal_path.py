import cv2
import numpy as np
from collections import deque

def generate_optimal_path(image_path, output="final_path.png"):
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w = img.shape[:2]
    # Use a smaller scale for faster pathfinding calculation
    scale = 0.3
    small = cv2.resize(img, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # --- ROAD DETECTION ---
    # Adjust these bounds if your path is a different color (e.g., darker asphalt)
    road_mask = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    road_mask[green_mask > 0] = 0

    # Clean the mask to connect small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_mask = cv2.dilate(road_mask, kernel, iterations=1)

    # --- DEFINE START AND END ---
    # Automatically finding the top-most and bottom-most road pixels
    ys, xs = np.where(road_mask > 0)
    if len(xs) == 0: return
    
    start = (xs[np.argmin(ys)], ys.min())
    end = (xs[np.argmax(ys)], ys.max())

    # --- PATHFINDING (BFS) ---
    def get_path(mask, start, end):
        queue = deque([(start, [start])])
        visited = set([start])
        while queue:
            (x, y), path = queue.popleft()
            if abs(x - end[0]) < 3 and abs(y - end[1]) < 3:
                return path
            # Search 8 neighbors
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < mask.shape[1] and 0 <= ny < mask.shape[0]:
                    if mask[ny, nx] > 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return None

    path_points = get_path(road_mask, start, end)

    # --- DRAW THE RED LINE ---
    if path_points:
        # Upscale points back to original image size
        scaled_points = [(int(p[0]/scale), int(p[1]/scale)) for p in path_points]
        
        # Draw the line smoothly
        for i in range(len(scaled_points) - 1):
            cv2.line(img, scaled_points[i], scaled_points[i+1], (0, 0, 255), 7, cv2.LINE_AA)

    cv2.imwrite(output, img)

# Usage: generate_optimal_path("input.jpg")