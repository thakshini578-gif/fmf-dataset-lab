import cv2, os
from datetime import datetime

ROOT, INCOMING = "fmfdataset", "incoming"
fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fc.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        gray = cv2.equalizeHist(gray)
        faces = fc.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
    return faces

for cls in ["imashi", "keshani"]:
    dst = os.path.join(ROOT, "raw", cls)
    for old in os.listdir(dst):
        os.remove(os.path.join(dst, old))
    saved = 0
    for f in sorted(os.listdir(os.path.join(INCOMING, cls))):
        img = cv2.imread(os.path.join(INCOMING, cls, f))
        if img is None:
            continue
        faces = detect(img)
        if len(faces) == 0:
            print(f"[{cls}] still no face in {f}")
            continue
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        pw, ph = int(0.2 * w), int(0.2 * h)
        crop = img[max(0, y-ph):min(img.shape[0], y+h+ph), max(0, x-pw):min(img.shape[1], x+w+pw)]
        saved += 1
        cv2.imwrite(os.path.join(dst, f"{cls}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{saved}.jpg"), crop)
    print(f"{cls}: saved {saved}")
