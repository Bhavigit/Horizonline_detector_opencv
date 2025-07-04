# Horizonline_detector_opencv

## 🖼 Sample Result

The script overlays a blue line across the image at the estimated horizon level.

##  Techniques Used

- OpenCV (cv2)
- Edge Detection (Canny)
- Hough Line Transform
- Geometric filtering based on line angle and position

## ▶ How to Run
1. Place your road images in a folder, e.g. `E:\test_images_horizon`
2. Update the `folder` variable in `horizon_detector.py` with the image folder path.
3. Run the script:

```bash
python horizon_detector.py
