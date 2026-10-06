import cv2, os, json
from datetime import datetime

ROOT = "fmfdataset"
INCOMING = "incoming"
class_names = json.load(open(f"{ROOT}/config.json"))["class_names"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for cls in class_names:
    src = os.path.join(INCOMING, cls)
    if not os.path.isdir(src):
        print(f"[{cls}] folder not found in incoming, skipped")
        continue
    dst = os.path.join(ROOT, "raw", cls)
    count = len(os.listdir(dst))
    saved = 0
    for f in sorted(os.listdir(src)):
        img = cv2.imread(os.path.join(src, f))
        if img is None:
            print(f"[{cls}] could not read {f}, skipped")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            print(f"[{cls}] no face detected in {f}, skipped")
            continue
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pw, ph = int(0.2 * w), int(0.2 * h)
        crop = img[max(0, y - ph):min(img.shape[0], y + h + ph),
                   max(0, x - pw):min(img.shape[1], x + w + pw)]
        count += 1
        saved += 1
        name = f"{cls}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{count}.jpg"
        cv2.imwrite(os.path.join(dst, name), crop)
    print(f"{cls}: saved {saved} cropped faces")
