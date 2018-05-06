# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image8_1]: ./output_8_1.png "Training Label Distribution"
[image12_0]: ./output_12_0.png "Gray image"
[image11]: ./new_signs/1.png "Visualization"
[image12]: ./new_signs/2.png "Grayscaling"
[image13]: ./new_signs/3.png "Random Noise"
[image14]: ./new_signs/4.png "Traffic Sign 1"
[image15]: ./new_signs/5.png "Traffic Sign 2"
[image16]: ./new_signs/6.png "Traffic Sign 3"
[image17]: ./new_signs/7.png "Traffic Sign 4"
[image18]: ./new_signs/8.png "Traffic Sign 5"
[image20]: ./output_26_0.png "Visualization"
[image21]: ./output_26_1.png "Visualization"
[image22]: ./output_26_2.png "Visualization"
[image23]: ./output_26_3.png "Visualization"
[image24]: ./output_26_4.png "Visualization"
[image25]: ./output_26_5.png "Visualization"
[image26]: ./output_26_6.png "Visualization"
[image27]: ./output_26_7.png "Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/liuxin00738/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images for each label in the trainning set.

![alt text][image8_1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because according to the paper gray scale has better performance. In addition, this will reduce the computational load during training.

Here is an example of a traffic sign image after grayscaling.

![alt text][image12_0]

As a last step, I normalized the image data because this will make the optimization algorithm converge easily and faster.

I did not generate additinoal data using distortion, etc. I tried direclty with the given data and it seems working fine.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|Index| Layer         		|     Description	        					| 
|:----------:|:---------------------:|:---------------------------------------------:| 
|1| Input         		| 32x32x1 Gray image   							| 
|2| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x38 	|
|3| RELU					|												|
|4| Max pooling	      	| 2x2 stride,  outputs 14x14x38 				|
|5| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
|6| RELU					|												|
|7| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
|8| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
|9| Dropout		| input is flattend layer7 and layer8( 2000x1)|
|10| Fully connected		|output is 43				|
|10| Softmax				|convert the digits using softmax 									| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer, with a batch size of 100, 10 epochs, and a learning rate of 0.0007. For the initialization of weights and bias, I use a truncated normal distribution with mean of 0 and standard deviation of 0.1. I trained with my own GPU(Quad M1200) and it is about 4 times faster than my CPU (i7_7820HQ). I tried more epochs, but 10 epoch is good enough.

Other hyper parameters include layer sizes. For the first and second convolution layers I choose size of 38 and 64, which is from the paper. For the third convolution layer, I choose a size of 400, which is from the LeNet lab. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.6% 
* test set accuracy of 94.3%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first model I choose is the LeNet from the lab of the course. It give a accuracy around 86%. This is choosen to have a runnable setup.

* What were some problems with the initial architecture?

The accuracy can't reach the 93% required by the project. Possibly this is due to that the size of the network is not deep enough.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I tried first to increase the size of the layers on the original LeNet model, but I am not able to increase the accuracy rate at the validating set above 93%. So I refer to the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" to adjust the structure of the network:
feed the first convolution layer also to the third convolution layer
increase the size of the layers
add dropout layer to increase robustness and prevent overfit
With these modifications I saw an improvement of accuracy rate. Then I tried the test set and it works.

* Which parameters were tuned? How were they adjusted and why?

Layer size. I increased the size to see their effect on the accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution layer is choosen for the spacital relation in the image is import. Also this will reduce the number of parameters. Dropout is added because this will increase the robustness and reduce the chance of overfit.

If a well known architecture was chosen:
* What architecture was chosen?

The architecture from "Traffic Sign Recognition with Multi-Scale Convolutional Networks" is choosen.

* Why did you believe it would be relevant to the traffic sign application?

It is the same problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The model has similar performance on the validation and test set, and I only test on the test set once (to prevent feedback manually the test set data to the model).

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14] ![alt text][image15] ![alt text][image16]
![alt text][image17] ![alt text][image18]

These images might be difficult to recognize due to the low resolution of the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction, the left image is the input, and the right image is the preicated output

![alt text][image20]

![alt text][image21]

![alt text][image22] 

![alt text][image23] 

![alt text][image24] 

![alt text][image25] 

![alt text][image26] 

![alt text][image27] 

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the new images. This is because relatively these images are in a better quality than some images from the test set. Also, I only use 10 training epoches, so the chance of overfit is smaller.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is in the python notebook. The top 5 prediction result is given in the jupyter notebook. For all the input images, the model is pretty certain of the result (all with a probability bigger than 96%). This means the model is quite confidence of the output.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


