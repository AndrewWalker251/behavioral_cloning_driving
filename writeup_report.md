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

[image1]: ./examples/center.jpg "center"
[image2]: ./examples/left.jpg "left"
[image3]: ./examples/right.jpg "right"

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md 

Reference - Used code from the course content and the YouTube Q and A https://www.youtube.com/watch?v=rpxZ87YFg0M&feature=youtu.be.

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

I added one layer of dropout after the first fully connected layer to help manage overfitting but this made the performance on the track worse so it was removed.

The main test of the model was to run it through the simulator and ensure that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and left and right camera data with a steering correction of 0.2 and -0.2 applied. This was sufficient. 

The batch size was 32 although as I used center, left and right images each batch included 96 images.

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

The following images show data from the training set and most importantly the difference between the center, left and right camera. 
![alt text][image1]
![alt text][image2]
![alt text][image3]

Although other augumentation options were available I stuck to a simple, normalise, use left center and right camera and a shuffle. 

I also cropped the image to get rid of unrequired data. 

After only 3 epochs the accuracy on the validation was sufficient to try on the simulator. 
