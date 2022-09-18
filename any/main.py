from any import Image, Camera, Face, Hand


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


if __name__ == '__main__':
    img()
    cam()
