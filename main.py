from my_modules import load_model, open_image, predict_box, apply_nms, torch_to_pil, plot_img_bbox, display_image, predict_text
import time

my_model = load_model('./Faster-RCNN-mobilenetv3.pth')

my_image = open_image('./test2.jpeg')

images = ['./test1.jpeg', './test2.jpeg', './test2.jpeg', './test4.jpeg']

for i in images:
	start = time.time()
	my_boxes = predict_box(i, my_model)
	end = time.time()
	print('time to predict', end - start)
	
	prediction, tensor_image = my_boxes[0], my_boxes[1]

	tensor_array = display_image(prediction, tensor_image)

	predict_text(tensor_image, tensor_array)
