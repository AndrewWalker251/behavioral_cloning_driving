# **Behavioral Cloning** 
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


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md 

#### 2. Functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code 

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture

For the final model architecture I chose to follow the Nvidia 'End to End Learning for Self Driving Cars' https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
This research is directly applicable to the same problem that we are trying to solve and so it seemed like an appropriate architechture to use.

As per the referenced Nvidia architecture:
* 3 convolution layers using (5x5) filters and a stride of 2. Each has a different number of filters (24, 36, 48)
* 2 convolution layers using (3x3) filters and a stride of 1. Each has 64 filters.
* Each convolutional layers used a relu activition to add non-linearity and aide speed of training.  
* Fully connected layers (100, 50 , 10 hidden nodes)
* Output single value for steering angle. 


#### 2. Overfitting

20% of the data was retained for validation. 

No additional dropout layers were required to increase the regularisation. 

The main test of the model was to run it through the simulator and ensure that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. tTraining data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and left and right camera data with a steering correction of 0.2 and -0.2 applied. This was sufficient. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to iterate through steadily more complicated set ups. 

I started with simple fully connected network. I then experimented getting the pipeline to work. This involved experimenting with the left and right camera and flipping images.
It also involved trying out different generators so that I would not get a memory error.

Eventually I only used the left and right camera (no flipping) and built a generator based on the code in the help section.

I used mean square error as a metric to guide training but mostly focused on the performance on the simulator.

Once I had this simple network working I then used the LeNet architecture. Although this performed well, the car kep leaving the track where the dirt off road joined and so this was not sufficient. 

Finally I implemented the Nvidia design as outlined above and the vehicle is able to drive autonomously around the track without leaving the road.


#### 3. Creation of the Training Set & Training Process

Although I played around with the simulator I found it quite difficult to drive and so used the pre-given training data as the basis to allow me to focus on the architecture experimentation. 

Although other augumentation options were available I stuck to a simple, normalise, use left center and right camera and a shuffle. 

After only 3 epochs the accuracy on the validation was sufficient to try on the simulator. 
