from any import Image, Camera, Face, Hand, Pose, SelfieSegmentation


def img():
    image = Image(r"wll-asl.jpg")
    hand = Hand(image)
    face = Face(image)
    image.draw_pints = hand.detect()
    image.draw_box = face.detect()
    image.show()


def cam():
    camera = Camera(0)
    camera.start()
    camera.toggle_show()
    hand = Hand(camera)
    face = Face(camera)
    while True:
        camera.draw_pints = hand.detect()
        camera.draw_box = face.detect()


def pos():
    image = Image(r"body.png")
    pose = Pose(image)
    image.draw_pints = pose.detect()
    image.show()


def seg():
    camera = Camera(0)
    camera.start()
    camera.toggle_show()
    i = Image("bg.jpg")
    ss = SelfieSegmentation(camera, i)
    while True:
        camera.replace = ss.detect()


if __name__ == '__main__':
    img()
    cam()
    pos()
    seg()
