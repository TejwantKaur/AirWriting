import time
import av
import cv2
import numpy as np

from pathlib import Path
from streamlit_webrtc import VideoProcessorBase

from src.hand_tracking_module import handDetector
from src.canvas_utils import (
    recognize_character,
    canvas_has_content,
)
from src.predict import (
    load_model,
    get_class_names,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MNIST_MODEL_PATH = (
    MODELS / "mnist_cnn.pth"
)

EMNIST_MODEL_PATH = (
    MODELS / "emnist_letters_cnn.pth"
)


class KidsWritingProcessor(VideoProcessorBase):

    def __init__(
        self,
        target_character,
        dataset_name,
    ):

        self.target_character = (
            target_character
        )

        self.dataset_name = dataset_name

        self.detector = handDetector(
            num_hands=1
        )

        self.xp = 0
        self.yp = 0

        self.img_canvas = None
        self.brush_thickness = 10
        self.draw_color = (255,0,255)
        
        self.last_draw_time = None

        self.result_message = ""
        self.last_prediction = ""


        if dataset_name == "mnist":
            model_path = (
                MNIST_MODEL_PATH
            )
        else:
            model_path = (
                EMNIST_MODEL_PATH
            )

        self.model = load_model(
            str(model_path),
            dataset_name=dataset_name
        )

        self.class_names = get_class_names(
            dataset_name
        )

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            if self.img_canvas is None:
                self.img_canvas = np.zeros(
                    (h, w, 3),
                    np.uint8
                )

            img = self.detector.findHands(img)
            lm_list = self.detector.findPosition(
                img,
                draw=False
            )

            if lm_list:

                x1, y1 = lm_list[8][1:]

                fingers = (
                    self.detector.fingersUp()
                )

                # DRAW
                if fingers == [0,1,0,0,0]:

                    self.last_draw_time = (
                        time.time()
                    )

                    if self.xp == 0 and self.yp == 0:

                        self.xp, self.yp = x1, y1

                    cv2.line(
                        img,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.draw_color,
                        self.brush_thickness
                    )

                    cv2.line(
                        self.img_canvas,
                        (self.xp, self.yp),
                        (x1, y1),
                        self.draw_color,
                        self.brush_thickness
                    )

                    self.xp, self.yp = x1, y1

                elif fingers == [1, 1, 1, 1, 1]:
                    erase_x, erase_y = (
                        lm_list[9][1],
                        lm_list[9][2]
                    )
                    cv2.circle(
                        self.img_canvas,
                        (erase_x, erase_y),
                        100,
                        (0, 0, 0),
                        cv2.FILLED
                    )
                else:
                    self.xp, self.yp = 0, 0

            img_gray = cv2.cvtColor(self.img_canvas,cv2.COLOR_BGR2GRAY)

            _, img_inv = cv2.threshold(
                img_gray,
                50,
                255,
                cv2.THRESH_BINARY_INV
            )

            img_inv = cv2.cvtColor(
                img_inv,
                cv2.COLOR_GRAY2BGR
            )

            # REMOVE DRAWING AREA
            img = cv2.bitwise_and(
                img,
                img_inv
            )

            # ADD CLEAN CANVAS
            img = cv2.bitwise_or(
                img,
                self.img_canvas
            )

            # ==================================
            # AUTO CHECK
            # ==================================

            
            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )
        
        except Exception as e:
            print("ERROR:", e)
            return frame
        

    def check_answer(self):
        if not canvas_has_content(
            self.img_canvas
        ):
            self.result_message = (
                "Write something first!"
            )
            return
        prediction, confidence = (
            recognize_character(
                self.img_canvas,
                self.model,
                self.class_names
            )
        )

        prediction = prediction.strip()
        self.last_prediction = prediction
        if (
            prediction.upper()
            ==
            self.target_character.upper()
        ):
            self.result_message = (
                f"🎉 Well done! "
                f"You wrote {prediction} correctly!"
            )
        else:
            self.result_message = (
                f"😊 You wrote {prediction}. "
                f"Try again!"
            )


    def clear_canvas(self):
        if self.img_canvas is not None:
            self.img_canvas[:] = 0
        self.xp, self.yp = 0, 0