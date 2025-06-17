import cv2 as cv
import numpy as np

# Set your HSV range
lower = np.array([0, 82, 147])
upper = np.array([23, 238, 255])

# Open webcam or video
cap = cv.VideoCapture("vids/volleyball_match.mp4")  # Or replace with 'volleyball_video.mp4'
if not cap.isOpened():
    print("Could not open video. Check the path.")
    exit()
fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize if needed
    frame = cv.resize(frame, (640, 480))

    # Convert to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Apply color mask
    mask = cv.inRange(hsv, lower, upper)

    # Optional: Remove noise
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 100:
            continue
        if area > 500:
            continue
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))

        if 0.7 < circularity < 1.2:
            x, y, w, h = cv.boundingRect(cnt)

    # Define object and surrounding regions (add margin)
            margin = 10
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, frame.shape[1])
            y2 = min(y + h + margin, frame.shape[0])

            object_roi = hsv[y:y + h, x:x + w]
            surrounding_roi = hsv[y1:y2, x1:x2]

            # Extract hue channel
            object_hue = object_roi[:, :, 0].flatten()
            surrounding_hue = surrounding_roi[:, :, 0].flatten()

            # Remove overlapping pixels (to get pure surrounding)
            surrounding_only = np.setdiff1d(surrounding_hue, object_hue)

            if len(surrounding_only) == 0:
                continue

            # Compute mean hue difference
            object_mean_hue = np.mean(object_hue)
            surrounding_mean_hue = np.mean(surrounding_only)
            hue_diff = abs(object_mean_hue - surrounding_mean_hue)

            # Filter based on hue contrast threshold
            if hue_diff > 20:  # Adjust threshold experimentally
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame, "Ball", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv.imshow("Ball Tracking", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
