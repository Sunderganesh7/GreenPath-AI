import cv2

def calculate_tree_distribution(image_path, boxes):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    dist = {
        "left": 0, "center": 0, "right": 0,
        "top": 0, "middle": 0, "bottom": 0
    }

    for (x1, y1, x2, y2) in boxes:
        # ✅ center of bounding box (PIXEL COORDS)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # ---------- HORIZONTAL ----------
        if cx < w / 3:
            dist["left"] += 1
        elif cx < 2 * w / 3:
            dist["center"] += 1
        else:
            dist["right"] += 1

        # ---------- VERTICAL ----------
        if cy < h / 3:
            dist["top"] += 1
        elif cy < 2 * h / 3:
            dist["middle"] += 1
        else:
            dist["bottom"] += 1

    return dist
