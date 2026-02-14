import cv2
import numpy as np
import os

def preprocessing_pipeline(image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)

    # 1️⃣ Resize
    image = cv2.resize(image, (512, 512))

    # 2️⃣ Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Preprocessed (blur)
    preprocessed = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4️⃣ Predicted tree mask (simple threshold)
    _, tree_mask = cv2.threshold(
        preprocessed,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Save all steps
    cv2.imwrite(os.path.join(output_dir, "pipeline_original.png"), image)
    cv2.imwrite(os.path.join(output_dir, "pipeline_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "pipeline_preprocessed.png"), preprocessed)
    cv2.imwrite(os.path.join(output_dir, "pipeline_tree_mask.png"), tree_mask)

    return True
