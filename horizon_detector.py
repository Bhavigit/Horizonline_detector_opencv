import cv2
import numpy as np
import math
import os

def detect_horizon_line(image_path, display_results=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Couldn't load image from {image_path}")
        return None

    original_img = img.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if display_results:
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Blurred", blurred)
        cv2.waitKey(1)

    edges = cv2.Canny(blurred, 50, 150)

    if display_results:
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=w // 4, maxLineGap=20)

    if lines is None:
        print("No lines detected.")
        return original_img

    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            angle_deg = 90
        else:
            angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

        if 0 <= angle_deg <= 10:
            mid_y = (y1 + y2) / 2
            if h * 0.25 <= mid_y <= h * 0.75:
                candidates.append(line[0])

    if not candidates:
        print("No valid horizon line candidates.")
        return original_img

    best_line = None
    max_length = 0
    for line in candidates:
        x1, y1, x2, y2 = line
        length = math.hypot(x2 - x1, y2 - y1)
        if length > max_length:
            max_length = length
            best_line = line

    if best_line is None:
        return original_img

    x1, y1, x2, y2 = best_line
    if x2 == x1:
        line_x1, line_y1 = x1, 0
        line_x2, line_y2 = x1, h
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        line_x1 = 0
        line_y1 = int(intercept)
        line_x2 = w - 1
        line_y2 = int(slope * (w - 1) + intercept)

    cv2.line(original_img, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 2)

    return original_img


if __name__ == "__main__":
    folder = r"E:\test_images_horizon"
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        print("No images found.")
    else:
        print(f"Found {len(image_files)} images.")

    for path in sorted(image_files):
        print(f"\nProcessing: {path}")
        result = detect_horizon_line(path, display_results=True)
        if result is not None:
            cv2.imshow(f"Output - {os.path.basename(path)}", result)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
