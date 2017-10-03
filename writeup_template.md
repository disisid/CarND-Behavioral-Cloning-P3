#**Behavioral Cloning Project Writeup** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_new00.h5 containing a trained convolution neural network 
* writeup_report.md

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_new00.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. NVIDIA Model 

Earlier I had tried implementing a basic CNN model as discussed in the class. It worked fairly well. However NVIDIA model as discussed in class worked way better
The model consists of a normailisation layer followed by 5 CNN layers and then three fully connected layers and finally a single output fully connected layer. 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

Initially the training was done on self captured data (using keyboard) howeverthe results were not enticing. Then I combined the training data captured using joystick alongwith the data provided by Udacity team. 

The model was trained and validated on different data sets ( the data set was split for validation and training) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. However a correction parameter for adjusting the steering angel for left camera and right images was used. Initially I used 0.2 however the car wavered off the path at one point (second curve which has two path emerging on track 1) but the car moved through majority of the track in the center of the track. 
Later on the correction paramter was tuned to 0.3. Moved across entire path on track but kept waving from one side of the road to other side. I would like to further tune this parameter later and come back on this project when time permits. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. I used augmentation to increase the dataset by left to right flipping the images. The NVIDIA paper also mentions of converting from RGB to YUV space - I would like to implement that later. 
Further in the training data captured, I went on opposite direction of the track and completed two laps.
Further cropping of images was used from top and bottom to include mainly the region of interest. 

####5. Use of Generators
I used generators as discussed in class. The generator code worked fine, however the training models went till 3 epochs. The loss continued to decrease as seen in figure 1. 

![alt text](https://github.com/disisid/CarND-Behavioral-Cloning-P3/blob/master/Figure_1.png "Figure 1 MSE vs Epochs")

However when I tested the same model but without generators, the val loss decreased only till epoch 1 and then started rising. So I used the first model checkpoint to run the final drive. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
