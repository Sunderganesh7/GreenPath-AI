import cv2
import numpy as np
import heapq

def generate_optimal_path(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tree areas as obstacles
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Cost map (trees = high cost)
    cost_map = np.where(mask == 255, 10, 1)

    start = (10, 10)
    end = (500, 500)

    path = dijkstra(cost_map, start, end)

    # Draw path
    for y, x in path:
        img[y, x] = (0, 0, 255)

    cv2.imwrite(output_path, img)


def dijkstra(cost_map, start, end):
    h, w = cost_map.shape
    dist = np.full((h, w), np.inf)
    dist[start] = 0
    prev = {}

    pq = [(0, start)]

    while pq:
        d, (y, x) = heapq.heappop(pq)

        if (y, x) == end:
            break

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                nd = d + cost_map[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[(ny, nx)] = (y, x)
                    heapq.heappush(pq, (nd, (ny, nx)))

    path = []
    cur = end
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()

    return path
