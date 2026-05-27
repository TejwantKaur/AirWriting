import time
import av
import cv2
import numpy as np

from pathlib import Path
from streamlit_webrtc import VideoProcessorBase

from src.hand_tracking_module import handDetector
from src.canvas_utils import recognize_character, canvas_has_content
from src.predict import load_model, get_class_names

from src.spell_correct import correct_word, get_correction_suggestions
from src.next_word import get_next_word_suggestions


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EMNIST_MODEL_PATH = MODELS / "emnist_letters_cnn.pth"


# LOAD MODEL
alphabet_model = load_model(
    str(EMNIST_MODEL_PATH),
    dataset_name="emnist_letters"
)

alphabet_classes = get_class_names(
    "emnist_letters"
)

DATA_PATH = ROOT / "data" / "entertainment_training_data.txt"

class AirWritingProcessor(VideoProcessorBase):
    def __init__(self):

        # HAND DETECTOR
        self.detector = handDetector(num_hands=1)

        # DRAWING STATE
        self.xp = 0
        self.yp = 0

        self.img_canvas = None
        self.brush_thickness = 8
        self.draw_color = (255, 0, 255)

        # PREDICTION STATE
        self.last_draw_time = None

        self.current_word = ""
        self.current_confidence = 0.0
        self.last_prediction = "Waiting for drawing..."
        self.last_raw_word = ""
        self.last_suggested_words = []
        self.next_word_suggestions = []
        self.suggestion_text = "No suggestions available"
        self.current_sentence = []
        self.instruction_text = (
            "Draw: index finger | Erase: open hand | "
            "Click Predict in the TV panel when ready"
        )

    def clear_canvas(self):
        if self.img_canvas is not None:
            self.img_canvas[:] = 0
        self.xp, self.yp = 0, 0
        self.last_prediction = "Canvas cleared"
        self.last_draw_time = None

    def reset_sentence(self):
        self.current_sentence = []
        self.last_raw_word = ""
        self.last_suggested_words = []
        self.next_word_suggestions = []
        self.last_prediction = "Sentence reset"
        self.suggestion_text = "No suggestions available"

    def delete_previous_word(self):
        if self.current_sentence:
            removed_word = self.current_sentence.pop()
            self.last_prediction = f"Removed: {removed_word}"
            self.last_raw_word = ""
            self.last_suggested_words = []
            self.next_word_suggestions = get_next_word_suggestions(
                " ".join(self.current_sentence)
            )

    def apply_correction(self, index):
        if 0 <= index < len(self.last_suggested_words) and self.current_sentence:
            chosen_word = self.last_suggested_words[index]
            self.current_sentence[-1] = chosen_word
            self.last_prediction = (
                f"Corrected: {self.last_raw_word} -> {chosen_word}"
            )
            self.last_raw_word = ""
            self.last_suggested_words = []
            self.suggestion_text = ""
            self.next_word_suggestions = get_next_word_suggestions(
                " ".join(self.current_sentence)
            )

    def apply_next_word(self, index):
        if 0 <= index < len(self.next_word_suggestions):
            chosen_word = self.next_word_suggestions[index]
            self.current_sentence.append(chosen_word)
            self.last_prediction = f"Added: {chosen_word}"
            self.last_raw_word = chosen_word
            self.last_suggested_words = get_correction_suggestions(chosen_word)
            self.next_word_suggestions = get_next_word_suggestions(
                " ".join(self.current_sentence)
            )

    def predict_canvas(self):
        if self.img_canvas is None or not canvas_has_content(self.img_canvas):
            self.last_prediction = "Canvas empty. Draw a word first."
            return

        word, confidence = recognize_character(
            self.img_canvas,
            alphabet_model,
            alphabet_classes
        )

        self.current_word = word
        self.current_confidence = confidence

        if word:
            self.last_raw_word = word
            self.current_sentence.append(word)

            self.last_suggested_words = [
                suggestion
                for suggestion in get_correction_suggestions(word)
                if suggestion.upper() != word.upper()
            ]

            self.next_word_suggestions = get_next_word_suggestions(
                " ".join(self.current_sentence)
            )

            if self.last_suggested_words:
                self.suggestion_text = " | ".join(
                    f"{idx + 1}:{text}"
                    for idx, text in enumerate(self.last_suggested_words)
                )
            else:
                self.suggestion_text = "No spelling suggestions"

            self.last_prediction = (
                f"Predicted: {word} ({confidence * 100:.2f}%)"
            )
        else:
            self.last_raw_word = ""
            self.last_suggested_words = []
            self.next_word_suggestions = []
            self.suggestion_text = "No suggestions available"
            self.last_prediction = "No word recognized"

        self.img_canvas[:] = 0
        self.xp, self.yp = 0, 0

    def recv(self, frame):
        try:
            # FRAME
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            # INIT CANVAS
            if self.img_canvas is None:
                self.img_canvas = np.zeros(
                    (h, w, 3),
                    np.uint8
                )

            # HAND TRACKING
            img = self.detector.findHands(img)
            lm_list = self.detector.findPosition(
                img,
                draw=False
            )

            # HAND GESTURES
            if lm_list:
                x1, y1 = lm_list[8][1:]
                fingers = self.detector.fingersUp()

                if fingers == [0, 1, 0, 0, 0]:
                    self.last_draw_time = time.time()
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1

                    # DRAW ON CAMERA
                    cv2.line(img,(self.xp, self.yp),(x1, y1),self.draw_color,self.brush_thickness)

                    # DRAW ON CANVAS
                    cv2.line(self.img_canvas,(self.xp, self.yp),(x1, y1),self.draw_color,self.brush_thickness)

                    self.xp, self.yp = x1, y1

                elif fingers == [1, 1, 1, 1, 1]:
                    erase_x, erase_y = (lm_list[9][1],lm_list[9][2])

                    cv2.circle(self.img_canvas,(erase_x, erase_y),100,(0, 0, 0),cv2.FILLED)

                else:
                    self.xp, self.yp = 0, 0

            # CANVAS MASKING
            img_gray = cv2.cvtColor(self.img_canvas,cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)

            img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

            # REMOVE DRAWING AREA
            img = cv2.bitwise_and(img,img_inv)

            # ADD CLEAN CANVAS
            img = cv2.bitwise_or(img,self.img_canvas)

            # DRAW IN-VIDEO STATUS TEXT
            # cv2.putText(img,self.instruction_text,(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2,cv2.LINE_AA,)
            cv2.putText(img,self.last_prediction,(20, 80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA,)

            if self.suggestion_text:
                cv2.putText(img,f"Suggestions: {self.suggestion_text}",
                    (20, 120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 200, 255),2,cv2.LINE_AA)

            if self.next_word_suggestions:
                next_text = " | ".join(
                    f"{idx + 4}:{word}"
                    for idx, word in enumerate(self.next_word_suggestions)
                )
                cv2.putText(img,f"Next: {next_text}",(20, 160),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 120, 0),2,cv2.LINE_AA,)

            if self.current_sentence:
                sentence_text = "Sentence: " + " ".join(self.current_sentence)
                cv2.putText(img, sentence_text,(20, 200),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2,cv2.LINE_AA,)

            # RETURN FRAME
            return av.VideoFrame.from_ndarray(img,format="bgr24")

        except Exception as e:
            print("VIDEO PROCESSOR ERROR:", e)
            return frame
        