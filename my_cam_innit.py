# #!/usr/bin/python3
# import time

# from picamera2 import Picamera2
# from picamera2.encoders import H264Encoder
# from picamera2.outputs import FfmpegOutput

# picam2 = Picamera2()
# video_config = picam2.create_video_configuration()
# picam2.configure(video_config)

# encoder = H264Encoder(10000000)
# output = FfmpegOutput('test.mp4', audio=True)

# picam2.start_recording(encoder, output)
# time.sleep(10)
# picam2.stop_recording()
'''


import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
picam2.configure(preview_config)

# picam2.start_preview(Preview.QTGL)

picam2.start()
# time.sleep(2)

metadata = picam2.capture_file("test.jpg")
print(metadata)

picam2.close()

'''

import numpy as np
from picamera2 import Picamera2, Preview
import cv2
import io

def take_photo_function():
    # Create a stream for image data
    picam2 = Picamera2()
    stream = io.BytesIO()
    preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
    picam2.configure(preview_config)
    # Adjust picam2 settings as needed
    picam2.resolution = (640, 480)
    picam2.start()
    picam2.capture(stream, format='jpeg')
    picam2.close()
    
    # Rewind the stream to the beginning so we can read its content
    stream.seek(0)
    
    # Convert the stream content to a numpy array
    image = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    
    # Decode the image array into an image matrix
    image = cv2.imdecode(image, 1)
    
    print(image)
take_photo_function()

