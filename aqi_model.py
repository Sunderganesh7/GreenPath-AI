import cv2
import numpy as np

def analyze_environment(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found"

    image = cv2.resize(image, (500, 500))

    # =============================
    # 1️⃣ VEGETATION DETECTION (GREEN + YELLOW)
    # =============================
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Include yellow-green to support autumn
    lower_veg = np.array([20, 40, 40])
    upper_veg = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_veg, upper_veg)

    vegetation_ratio = np.sum(mask > 0) / (image.shape[0] * image.shape[1])
    vegetation_score = vegetation_ratio * 100


    # =============================
    # 2️⃣ CLARITY / HAZE DETECTION
    # =============================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize clarity
    clarity_score = min(clarity / 180, 1) * 100


    # =============================
    # 3️⃣ BRIGHTNESS ANALYSIS
    # =============================
    brightness = np.mean(gray)

    # Penalize only extreme brightness (dust/smog)
    if brightness > 190:
        brightness_score = 50
    elif brightness < 60:
        brightness_score = 60
    else:
        brightness_score = 100


    # =============================
    # 4️⃣ CONTRAST CHECK
    # =============================
    contrast = gray.std()
    contrast_score = min(contrast / 70, 1) * 100


    # =============================
    # 5️⃣ COMBINED ENVIRONMENT SCORE
    # =============================

    final_score = (
        0.35 * vegetation_score +
        0.30 * clarity_score +
        0.20 * contrast_score +
        0.15 * brightness_score
    )

    final_score = int(final_score)


    # =============================
    # 6️⃣ AQI CLASSIFICATION
    # =============================

    if final_score >= 80:
        category = "Good (0-50) 🌿"
    elif final_score >= 60:
        category = "Moderate (51-100) 🌤"
    elif final_score >= 40:
        category = "Unhealthy (101-200) 🌫"
    else:
        category = "Poor (200+) ⚠"

    return f"Environmental Score: {final_score}/100 | Estimated AQI: {category}"
