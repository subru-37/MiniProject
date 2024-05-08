from my_modules import config, write, load_model, take_image, open_image, predict_box, apply_nms, torch_to_pil, plot_img_bbox, display_image, predict_text
import time
import multiprocessing
from sys import stdin
from termios import TCIOFLUSH, tcflush
from time import strftime
from pynput import keyboard
from picamera2 import Picamera2, Preview
from datetime import datetime

my_model = load_model('./models/Faster-RCNN-mobilenetv3.pth')

pressed_key = ''
cam = Picamera2()
cam.start_preview(Preview.QTGL)
cam.start()

db = config('./creds/creds.json')

def take_photo():
    # Capture a PNG image while still running in the preview mode.
    # The image release is activated with the button 'p' and with 'q' the script is stopped
    key_flag = False
    
    global pressed_key
    global listener
    my_handler = keyboard.Controller()

    try:
        while True:
            # print(pressed_key)
            time.sleep(1)
            print('Waiting to Capture Image')
            if pressed_key == 'c':
                    # filename = strftime("%Y%m%d-%H%M%S") + '.png'
                    filename = 'test.jpeg'
                    cam.capture_file(filename, format="jpeg", wait=None)
                    print(f"\rCaptured {filename} succesfully")
                    pressed_key = 'l'
                    my_image = open_image('./test.jpeg')
                    my_boxes = predict_box('./test1.jpeg', my_model)
                    
                    prediction, tensor_image = my_boxes[0], my_boxes[1]

                    tensor_array = display_image(prediction, tensor_image)

                    texts = predict_text(tensor_image, tensor_array)
                    current_time = datetime.now()
                    data = {"number_plate": texts, "timestamp": current_time}
                    write(db, "number_plates", data)

            if pressed_key == 'x':
                print("\rClosing camera...")
                break
    finally:
        cam.stop_preview()
        cam.stop()
        cam.close()
        tcflush(stdin, TCIOFLUSH)
        listener.stop()

def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        global pressed_key
        pressed_key = key.char
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    # global pressed_key
    # pressed_key = key.char
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()
take_photo()
