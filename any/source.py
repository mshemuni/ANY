import threading
import cv2


class Source:
    def __init__(self):
        """
        Base Source class which any other sources would inherit from
        """
        self.data = None


class Image(Source):
    def __init__(self, path: str) -> None:
        """
        Image class. Using opencv-python it will get image data from a file
        :param path: Image path
        """
        super().__init__()
        self.path = path
        self.data = cv2.imread(self.path)
        self.draw_box = None
        self.draw_pints = None
        self.replace = None

    def show(self) -> None:
        """
        Shows the image in question
        """
        # Draws boxes on the image if box list were given
        if self.draw_box is not None:
            for each in self.draw_box.to_numpy():
                each = each.astype(int)
                self.data = cv2.rectangle(self.data, (each[0], each[1]), (each[0] + each[3], each[1] + each[2]),
                                          (255, 0, 0), 2)
        # Draws points on the image if box list were given
        if self.draw_pints is not None:
            for each in self.draw_pints.to_numpy():
                each = each.astype(int)
                self.data = cv2.circle(self.data, (each[0], each[1]), 5, (0, 255, 0), -1)

        # Replace the image ith given image
        if self.replace is not None:
            self.data = self.replace

        cv2.imshow("image", self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Camera(threading.Thread, Source):
    def __init__(self, source_id: int) -> None:
        """
        Camera class. Usinhg opencv-python it will get image from a webcam

        :param source_id: Camera ID
        """
        threading.Thread.__init__(self)
        self.source_id = source_id
        self.vid = cv2.VideoCapture(self.source_id)
        self.show = False
        self.data = None
        self.draw_box = None
        self.draw_pints = None
        self.replace = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.source_id})"

    def toggle_show(self) -> None:
        """
        Toggles showing the image window
        """
        self.show = not self.show

    def run(self) -> None:
        """
        Starts a thread to get image from the camera and writes it to `self.data` to be accessed
        """
        ret, self.data = self.vid.read()
        while ret:
            ret, self.data = self.vid.read()
            # Show's the window if requested
            if self.show:
                # Draws boxes on the image if box list were given
                if self.draw_box is not None:
                    for each in self.draw_box.to_numpy():
                        each = each.astype(int)
                        self.data = cv2.rectangle(self.data, (each[0], each[1]), (each[0] + each[3], each[1] + each[2]),
                                                  (255, 0, 0), 2)
                # Draws points on the image if box list were given
                if self.draw_pints is not None:
                    for each in self.draw_pints.to_numpy():
                        each = each.astype(int)
                        self.data = cv2.circle(self.data, (each[0], each[1]), 5, (0, 255, 0), -1)

                # Replace the image ith given image
                if self.replace is not None:
                    cv2.imshow('frame', self.replace)
                else:
                    cv2.imshow('frame', self.data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.vid.release()
        cv2.destroyAllWindows()
