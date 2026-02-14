import cv2
import numpy as np
import os

def green_cover_estimation(image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Read image ----
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read")

    image = cv2.resize(image, (600, 400))

    # ---- Extract image name ----
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ---- Convert to HSV ----
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ---- Green color range ----
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # ---- Mask ----
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # ---- Percentages ----
    total_pixels = green_mask.size
    green_pixels = cv2.countNonZero(green_mask)
    non_green_pixels = total_pixels - green_pixels

    green_percent = (green_pixels / total_pixels) * 100
    non_green_percent = (non_green_pixels / total_pixels) * 100

    # =================================================
    # ✅ CORRECT VISUALIZATION (OVERLAY, NOT CUT-OUT)
    # =================================================

    # Green overlay
    green_overlay = image.copy()
    green_overlay[green_mask > 0] = (0, 255, 0)

    green_cover_img = cv2.addWeighted(
        image, 0.7,
        green_overlay, 0.3,
        0
    )

    # Non-green overlay
    non_green_overlay = image.copy()
    non_green_overlay[green_mask == 0] = (180, 180, 180)

    non_green_img = cv2.addWeighted(
        image, 0.7,
        non_green_overlay, 0.3,
        0
    )

    # ---- UNIQUE filenames ----
    green_path = f"{base_name}_green_cover.png"
    non_green_path = f"{base_name}_non_green_cover.png"

    cv2.imwrite(os.path.join(output_dir, green_path), green_cover_img)
    cv2.imwrite(os.path.join(output_dir, non_green_path), non_green_img)

    return (
        round(green_percent, 2),
        round(non_green_percent, 2),
        green_path,
        non_green_path
    )
