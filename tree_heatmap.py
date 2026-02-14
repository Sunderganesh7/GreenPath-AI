import cv2
import numpy as np

def create_tree_heatmap(image, boxes):
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Add tree centers
    for x1, y1, x2, y2 in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if 0 <= cx < w and 0 <= cy < h:
            heatmap[cy, cx] += 1

    # Smooth heatmap
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=25, sigmaY=25)

    # Normalize
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    # Color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    output = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

    return output
