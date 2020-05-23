# Vehicle Detection using TensorFlow Object Detection API by  fine-tuning pre-trained SSD+Inception V2 Model

------



# Introduction

As a part of the self-driving cars technology, vehicle's perception is a challenging task for the  computer vision field. This project aims to build a computer vision algorithm to detect front and rear car views using the TensorFlow Object Detection API by fine-tuning pre-trained state-of-the-art SSD+Inception V2 Model trained on the COCO dataset.

------



## Training

- ### The Dataset

Davis King’s vehicles dataset to differentiate the front and rear views of the vehicles. The dataset is from Davis King’s dlib library and each image in the dataset is captured from a camera mounted to a car’s dashboard. For each image, all visible front and rear views of vehicles are labeled as such.

- ### Training performance

  ![]()

  ![]()

  

- ### Evaluation

  ![]()

  ![]()



## Prediction

- ### Apply to images and videos

The `predict_image.py`applys the network we trained to an input image outside the dataset it is trained on. And the `predict_video.py`apply to an input video.

The following command can apply SSD model to inference of images and videos.
```
python predict_image.py --model PATH_TO_DIR/fronzen_inference_graph.pb --labels PATH_TO_CLASSES_FILE/classes.pbtxt --image SAMPLE_IMAGE.jpg --num_classes NUM_OF_CLASSES
```
```
python predict_video.py --model PATH_TO_DIR/fronzen_inference_graph.pb --labels PATH_TO_DIR/classes.pbtxt --input PATH_TO_DIR/SAMPLE_VIDEO.mp4 --output PATH_TO_DIR/OUTPUT_VIDEO.mp4 --num_classes NUM_OF_CLASSES
```
