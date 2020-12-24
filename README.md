# Emotion-Detection

Detecting emotions of a person in real-time using deep learning. This project was made using Tensorflow and OpenCV. Tried out several models like VGG19, ResNet50 and also created a new custom model.

## Installation and Setup
* Fork the repo and clone it.
```
git clone https://github.com/Frostday/Emotion-Detection.git
```
* To download the required packages run the command below 
```
pip install -r requirements.txt
```
* Run the following command to see the results
```
python run_v1.py
```
* You can view all model structures and training inside models_v1 directory

## Dataset(version 1)

The data used can be found at this link - https://github.com/muxspace/facial_expressions. When using the dataset, make sure that all categories should have roughly equal number of images otherwise our neural network won't be able to make accurate predictions and will instead favour the categories which had higher number of images during training.

## Models and Conclusion(version 1)

First, I tried fine tuning well known models like ResNet50 and VGG19. I got a significantly low loss and high accuracy but the models were not able to perform well in real-time. It almost felt like they were trained to recognize faces instead of expressions.<br><br>
Next, I tried creating a custom model which gave me a significantly higher loss and lower accuracy as compared to the previous two models but this model performed significantly better in real-time as compared to the last two models as it was able to detect a change in expression but it's accuracy still wasn't high enough to give a decent result.<br><br>
In my opinion, this is the result of over-training on a very small dataset which leads to the model focusing more on faces than their expression to get results. In order to fix this, a better dataset is required.

## Room for Improvement

* We can try using a better dataset.
* Try using more custom models and more hyperparameter tuning.

## Dataset(version 2)

The data used can be found at this link - https://anonfiles.com/bdj3tfoeba/data_zip. This is the FER-2013 dataset which contains 48x48 pixel grayscale images of faces divided between 7 classes. In total this dataset has 28,709 train images and 7178 test images but during training some images were removed to balance out the class distribution.

## Models and Conclusion(version 2)

First, I tried fine tuning well known models like ResNet50 and VGG19 but I got horrible results in terms of loss and accuracy. Next, I tried a custom model which gave me significantly better results as it got a lot better accuracy and lower loss than the pretrained models. This custom model was able to properly detect changes in my expressions this time but the accuracy was still too low to be used anywhere.