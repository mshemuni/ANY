from .source import Source
import cv2
import mediapipe as mp
import pandas as pd


class Detector:
    def __init__(self, source: Source) -> None:
        """
        Base Detector class which all other detectors would inherit from
        :param source: The source. An object where one can read image data from `source.data`
        """
        self.source = source

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(Source={self.source})"


class Hand(Detector):
    def __init__(self, source: Source) -> None:
        """
        The hand detector. Uses mediapipe
        """
        super().__init__(source)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=0.5
        )

    def detect(self) -> pd.DataFrame:
        """
        detection
        :return: points of hands
        """
        points = []
        if self.source.data is not None:
            results = self.hands.process(cv2.cvtColor(self.source.data, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                points = [
                    [each.x * self.source.data.shape[1], each.y * self.source.data.shape[0]]
                    for hand in results.multi_hand_landmarks
                    for each in hand.landmark
                ]
        return pd.DataFrame(
            points, columns=["x", "y"]
        )


class Face(Detector):
    def __init__(self, source) -> None:
        """
        The face detector. Uses mediapipe
        """
        super().__init__(source)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    def detect(self) -> pd.DataFrame:
        """
        detection
        :return: boxes of faces
        """
        boxes = []
        if self.source.data is not None:
            results = self.face_detection.process(cv2.cvtColor(self.source.data, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    boxes.append(
                        [
                            box.xmin * self.source.data.shape[1],
                            box.ymin * self.source.data.shape[0],
                            box.width * self.source.data.shape[1],
                            box.height * self.source.data.shape[0]
                        ]
                    )

        return pd.DataFrame(
            boxes, columns=["x", "y", "w", "h"]
        )
