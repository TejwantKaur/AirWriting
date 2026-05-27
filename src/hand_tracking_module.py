import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class handDetector():
    def __init__(self, model_path="models/hand_landmarker.task", num_hands=2):
        BaseOptions = python.BaseOptions

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=num_hands
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.tipIds = [4, 8, 12, 16, 20]

        # manual connections
        self.HAND_CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20)
        ]

    def findHands(self, img, draw=True):
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.hand_landmarker.detect(mp_image)

        h, w, c = img.shape

        if self.results.hand_landmarks:
            for hand in self.results.hand_landmarks:
                # draw points
                for id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                # draw connections
                if draw:
                    for connection in self.HAND_CONNECTIONS:
                        x1 = int(hand[connection[0]].x * w)
                        y1 = int(hand[connection[0]].y * h)
                        x2 = int(hand[connection[1]].x * w)
                        y2 = int(hand[connection[1]].y * h)

                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.hand_landmarks and len(self.results.hand_landmarks) > handNo:
            myHand = self.results.hand_landmarks[handNo]

            h, w, c = img.shape

            for id, lm in enumerate(myHand):
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return self.lmList  

    def fingersUp(self):
        if len(self.lmList) == 0:
            return []

        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]: 
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers