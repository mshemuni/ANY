from .source import Source, Image
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


class Detector:
    def __init__(self, source: Source) -> None:
        """
        Base Detector class which any other detectors would inherit from
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


class Pose(Detector):
    def __init__(self, source: Source) -> None:
        """
        The pose detector. Uses mediapipe
        """
        super().__init__(source)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )

    def detect(self) -> pd.DataFrame:
        """
        detection
        :return: points of hands
        """
        points = []
        if self.source.data is not None:
            results = self.pose.process(cv2.cvtColor(self.source.data, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                points = [
                    [each.x * self.source.data.shape[1], each.y * self.source.data.shape[0]]
                    for each in results.pose_landmarks.landmark
                ]
        return pd.DataFrame(
            points, columns=["x", "y"]
        )


class Face(Detector):
    def __init__(self, source: Source) -> None:
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


class SelfieSegmentation(Detector):
    def __init__(self, source: Source, image: Image) -> None:
        """
        The Selfie Segmentation detector. Uses mediapipe
        """
        super().__init__(source)
        self.image = image
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation

        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    def detect(self) -> np.ndarray:
        """
        detection
        :return: new_image
        """
        if self.source.data is not None:
            bg = self.image.data.copy()
            image = cv2.cvtColor(self.source.data, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False
            results = self.selfie_segmentation.process(image)
            # image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            new_image = image.copy()
            bg = cv2.resize(bg, (new_image.shape[1], new_image.shape[0]), interpolation=cv2.INTER_AREA)
            bg[condition] = new_image[condition]
            return bg.astype(np.uint8)
