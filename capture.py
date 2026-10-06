import cv2, os, sys, json
from datetime import datetime

ROOT = "fmfdataset"
class_names = json.load(open(f"{ROOT}/config.json"))["class_names"]

if len(sys.argv) < 2 or sys.argv[1] not in class_names:
    print("Usage: python capture.py <classname>   options:", class_names)
    sys.exit()

cls = sys.argv[1]
save_dir = os.path.join(ROOT, "raw", cls)
os.makedirs(save_dir, exist_ok=True)
count = len(os.listdir(save_dir))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    display = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(display, f"Class: {cls} | Photos: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if len(faces) == 0:
        cv2.putText(display, "No face detected", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display, "SPACE=capture  q=quit", (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # spacebar
        if len(faces) == 0:
            print("No face detected, try again")
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_w, pad_h = int(0.2 * w), int(0.2 * h)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(frame.shape[1], x + w + pad_w), min(frame.shape[0], y + h + pad_h)
        crop = frame[y1:y2, x1:x2]

        count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{cls}_{ts}_{count}.jpg"
        cv2.imwrite(os.path.join(save_dir, fname), crop)
        print("Saved", fname)

cap.release()
cv2.destroyAllWindows()
