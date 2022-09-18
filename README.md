# ANY
ANY (`niAei visioN librarY`) is a small library using mediapipe and opencv to manage detection.

It can handle `Face` and `Hand` detection. It consists of two parts.

- `Source`
- `Detector`

## `Source`
Source is referred to collection of objects such as `Camera` or `Image`. This kind of objects can be expended only
must for a `Source` object is that the image data must be stored in `source.data`. So other objects can read image data
and do operations.

- `Image`
- `Camera`

### `Image`
`Image` can read data from disk and stores the data in `self.data`.

```python
from any import Image

image = Image(r"wll-asl.jpg")
image.show()
```

### `Camera`
Simular to `Image`, `Camera` stores the data in `self.data`, the difference is that Camera will read from a device and not 
disk.

```python
from any import Camera

camera = Camera(0)
camera.start()
camera.toggle_show()
```

## Detector
Detector is referred to collection of objects such as `Hand` or `Face`. This kind of objects can be expended only
must for a `Detector` object is that it must contain a `detect` method. This method must return `pandas` `DataFrame`.

- `Hand`
- `Face`
- `Pose`

### Hand
The `Hand` class uses `mediapipe` and can detect hands on a source.

```python
from any import Image, Hand

image = Image(r"wll-asl.jpg")
hand = Hand(image)
image.draw_pints = hand.detect()
image.show()
```

Here we can see the hand detector was executed on an `Image` source.

![Hand](hand.png)

### Face
The `Face` class uses `mediapipe` and can detect faces on a source.

```python
from any import Camera, Face

camera = Camera(0)
camera.start()
camera.toggle_show()
face = Face(camera)
while True:
    camera.draw_box = face.detect()
```

Here we can see the face detector was executed on a `Camera` source.

![Face](face.png)

### Pose
The `Pose` class uses `mediapipe` and can detect poses on a source.

```python
from any import Image, Pose

image = Image(r"body.png")
pose = Pose(image)
image.draw_pints = pose.detect()
image.show()
```

Here we can see the body detector was executed on a `Image` source.

![Body](body.png)

### SelfieSegmentation
The `SelfieSegmentation` class uses `mediapipe` and can detect person and return a mask. One can use the mask to replace background.

```python
from any import Camera, Image, SelfieSegmentation

camera = Camera(0)
camera.start()
camera.toggle_show()
i = Image("bg.jpg")
ss = SelfieSegmentation(camera, i)
while True:
    camera.replace = ss.detect()
```

Here we can see the SelfieSegmentation was executed on a `Camera` source and background was replaced by an `Image`.

![SelfieSegmentation](segmantation.gif)

## Getting `Any`
`Any` is open source and can be used as GPL-3.0 license allows

### Download
`Any` can be downloaded as:

```bash
git clone https://github.com/mshemuni/ANY.git

cd ANY
```

### Installation
`Any` has requirements such as:
- `mediapipe==0.8.11`
- `opencv_python==4.6.0.66`
- `pandas==1.4.4`
- `numpy~=1.23.3`

To install:

```bash
pip install mediapipe==0.8.11 opencv_python==4.6.0.66 pandas==1.4.4 numpy~=1.23.3
```

or

```bash
pip install -r requirements.txt
```

### Usage
Run the `main.py` to see the results.

```bash
python main.py
```

## Author

Mohammad S.Niaei

m.shemuni@gmail.com
