import cv2
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMNIST_MODEL_PATH = ROOT / "models" / "emnist_letters_cnn.pth"
MNIST_MODEL_PATH = ROOT / "models" / "mnist_cnn.pth"
RECOGNITION_MODES = {
    "1": ("mnist", MNIST_MODEL_PATH, "Digit"),
    "2": ("emnist_letters", EMNIST_MODEL_PATH, "Text"),
}

def prepare_canvas_for_model(crop):
    ys, xs = np.where(crop > 0)
    if len(xs) and len(ys):
        crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    height, width = crop.shape[:2]
    side = max(height, width) + 16
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    square[y_offset:y_offset + height, x_offset:x_offset + width] = crop

    gray = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = torch.tensor(gray, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def split_wide_contour(thresh, rect):
    x, y, w, h = rect
    if w < max(35, int(h * 1.7)):
        return [rect]

    crop = thresh[y:y + h, x:x + w]
    estimated_chars = int(np.clip(round(w / max(h * 0.65, 1)), 2, 12))
    projection = crop.sum(axis=0).astype(np.float32)
    window = max(3, w // 35)
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(projection, kernel, mode="same")

    cuts = [0]
    char_width = w / estimated_chars
    for index in range(1, estimated_chars):
        center = int(round(index * char_width))
        radius = max(4, int(char_width * 0.35))
        left = max(cuts[-1] + 5, center - radius)
        right = min(w - 5, center + radius)
        if left >= right:
            continue
        cuts.append(left + int(np.argmin(smoothed[left:right])))
    cuts.append(w)

    segments = []
    for left, right in zip(cuts, cuts[1:]):
        segment = crop[:, left:right]
        ys, xs = np.where(segment > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        sx = x + left + xs.min()
        sy = y + ys.min()
        sw = xs.max() - xs.min() + 1
        sh = ys.max() - ys.min() + 1
        if sw >= 4 and sh >= 10:
            segments.append((sx, sy, sw, sh))

    return segments or [rect]


def recognize_character(img_canvas, model, class_names):
    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if rect[2] >= 10 and rect[3] >= 10:
            boxes.extend(split_wide_contour(thresh, rect))

    boxes = sorted(boxes, key=lambda rect: rect[0])

    if not boxes:
        print("Canvas is empty")
        return "", 0.0

    result = ""
    confidences = []
    pad = 20

    for x, y, w, h in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_canvas.shape[1], x + w + pad)
        y2 = min(img_canvas.shape[0], y + h + pad)

        crop = thresh[y1:y2, x1:x2]
        digit_tensor =  prepare_canvas_for_model(crop)

        with torch.no_grad():
            output = model(digit_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        label = class_names[predicted_class.item()]
        confidence = confidence.item()
        result += label
        confidences.append(confidence)

        print(f"Predicted: {label} Confidence: {confidence * 100:.2f}%")

    avg_confidence = sum(confidences) / len(confidences)
    return result, avg_confidence

def canvas_has_content(img_canvas):
    if img_canvas is None:
        return False
    gray = cv2.cvtColor(
        img_canvas,
        cv2.COLOR_BGR2GRAY
    )
    return np.count_nonzero(gray) > 50

# def main():
#     print("Choose recognition mode:")
#     print("1. Recognize digit")
#     print("2. Recognize text")
#     mode = input("Enter 1 or 2: ").strip() or "1"
#     dataset_name, model_path, prediction_label = RECOGNITION_MODES.get(
#         mode,
#         RECOGNITION_MODES["1"],
#     )

#     if not model_path.exists():
#         print(f"Missing model: {model_path}")
#         return

#     class_names = get_class_names(dataset_name)
#     model = load_model(str(model_path), dataset_name=dataset_name)
#     detector = htm.handDetector(num_hands=1)

#     cap = cv2.VideoCapture(0)
#     cap.set(3, WIDTH)
#     cap.set(4, HEIGHT)

#     img_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
#     xp, yp = 0, 0
#     recognized_text = ""
#     confidence = 0.0
#     show_result = False
#     result_timer = 0

#     try:
#         while True:
#             success, img = cap.read()
#             if not success:
#                 break

#             img = cv2.flip(img, 1)
#             img = detector.findHands(img)
#             lm_list = detector.findPosition(img, draw=False)

#             if lm_list:
#                 x1, y1 = lm_list[8][1:]
#                 fingers = detector.fingersUp()

#                 if fingers == [0, 1, 0, 0, 0]:
#                     cv2.circle(img, (x1, y1), 10, DRAW_COLOR, cv2.FILLED)

#                     if xp == 0 and yp == 0:
#                         xp, yp = x1, y1

#                     cv2.line(img, (xp, yp), (x1, y1), DRAW_COLOR, BRUSH_THICKNESS)
#                     cv2.line(img_canvas, (xp, yp), (x1, y1), DRAW_COLOR, BRUSH_THICKNESS)
#                     xp, yp = x1, y1
#                 else:
#                     xp, yp = 0, 0

#                 if fingers == [1, 1, 1, 1, 1]:
#                     erase_x, erase_y = lm_list[9][1], lm_list[9][2]
#                     cv2.circle(img_canvas, (erase_x, erase_y), 170, (0, 0, 0), cv2.FILLED)

#             img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
#             _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
#             img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
#             img = cv2.bitwise_and(img, img_inv)
#             img = cv2.bitwise_or(img, img_canvas)

#             cv2.putText(img, "Index finger: draw | S: predict | C: clear | ESC: exit",
#                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                         (0, 255, 255), 2)

#             if show_result and recognized_text:
#                 label = f"{prediction_label}: {recognized_text} ({confidence * 100:.1f}%)"
#                 cv2.rectangle(img, (15, 625), (600, 695), (0, 0, 0), cv2.FILLED)
#                 cv2.putText(img, label, (25, 675), cv2.FONT_HERSHEY_SIMPLEX,
#                             1.8, (0, 255, 0), 4)

#                 elapsed = cv2.getTickCount() - result_timer
#                 if elapsed > cv2.getTickFrequency() * 3:
#                     show_result = False

#             cv2.imshow("Air Canvas", img)
#             key = cv2.waitKey(1) & 0xFF

#             if key == 27:
#                 break

#             if key == ord("s"):
#                 recognized_text, confidence = recognize_character(img_canvas, model, class_names)
#                 show_result = True
#                 result_timer = cv2.getTickCount()
#                 img_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
#                 xp, yp = 0, 0

#             if key == ord("c"):
#                 img_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
#                 recognized_text = ""
#                 confidence = 0.0
#                 xp, yp = 0, 0
#                 print("Canvas cleared")

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
