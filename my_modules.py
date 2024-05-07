# Basic python and ML Libraries
import numpy as np
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from PIL import Image
import pytesseract
import easyocr
from picamera2 import Picamera2, Preview
import os



def load_model(name):
    # Load the entire model
    loaded_model = torch.load(name, map_location=torch.device('cpu'))
    return loaded_model

def take_image():
    picam2 = Picamera2()

    preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})
    picam2.configure(preview_config)

    # picam2.start_preview(Preview.QTGL)

    picam2.start()
    # time.sleep(2)

    metadata = picam2.capture_file("test.jpg")
    print(metadata)
    picam2.close()

def open_image(path):
    test = path
    image = cv2.imread(test)
    print(image.shape)
    return image

def predict_box(test, loaded_model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # loaded_model.eval()
    image_path = test
    image = Image.open(image_path)

    # Define the transformations
    transform = torchtrans.Compose([
        # torchtrans.Resize((256, 256)),  
        torchtrans.ToTensor()  
    ])

    # Apply the transformations
    tensor_image = transform(image)
    with torch.no_grad():
        prediction = loaded_model([tensor_image.to(device)])[0]

    return [prediction, tensor_image]

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    print(type(img))
    a.imshow(img)
    i=0
    for box in (target):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        image = np.array(img)
        cropped_image = image[int(y):int(y+height), int(x):int(x+width)]
        cv2.imwrite('./crops/cropped_image{0}.png'.format(i),cropped_image)
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')
        i+=1
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()

def display_image(prediction, tensor_image):
    nms_prediction = apply_nms(prediction, iou_thresh=0.2)
    print('NMS APPLIED MODEL OUTPUT')
    tensor_array = nms_prediction['boxes'].cpu()
    plot_img_bbox(torch_to_pil(tensor_image), tensor_array)
    print(tensor_array.numpy())
    return tensor_array


def predict_text(tensor_image, tensor_array):
    numpy_image = tensor_image.cpu().numpy()

    numpy_image = (numpy_image * 255).astype(np.uint8)

    numpy_image = np.transpose(numpy_image, (1, 2, 0))

    rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    print(rgb_image.shape)
    roi = tensor_array.numpy()
    for i in roi:
        x, y, w, h = i
        print(x,y,w,h)
        plate = rgb_image[int(y):int(h), int(x):int(w)]
        fig, a = plt.subplots(1,1)
        fig.set_size_inches(5,5)
        a.imshow(plate)
        reader = easyocr.Reader(['en'])  # English language
        result = reader.readtext(plate)
        text = pytesseract.image_to_string(plate, config='--psm 6')
        print('PyTesseract:',text)
        try: 
            print(result[0][1])
        except:
            print('platw not recognized')



