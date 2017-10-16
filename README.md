# Traffic-Sign-Classifier
**Build a Traffic Sign Recognition Project**

# Overview 
In this project we build a model which would enable the computer to classify traffic signs using CNN (Made using Tensorflow). 
# Steps
The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
# Example Data

![Set of images per class](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/Screenshot (24).png)


Here's the code for the project! [project code](https://github.com/Shreyas3108/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

To read the data which was in pickle format we use pickle library , Then the train data was split into validation set of 20% by using Sklearn library's train and test split function. 
Statistics for signs data set: 
* The size of training set is - 31367 
* The size of the validation set is - 7842 
* The size of test set is - 12630
* The shape of a traffic sign image is = (32,32,3) 
* The number of unique classes/labels in the data set is - 43 unique classes. 
The names for the classes are available at the signnames.csv from which we can map the names of images.

### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Train Dataset](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/train.png)
![Test Dataset](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/test.png)

## PreProcess Data

To preprocess the data we use [OpenCv](https://opencv.org) and convert image to grayscale from which we use the technique of normalization ie , a + ((img - minimum) * (b - a)) / (maximum - minimum) 
We can also optionally use :- img - mean(img) / max(img) - min(img).

Used greyscale technique since the traffic signs have least effect of colors and due to this minimal affect of colors , Classifier can also avoid false classification based on colors. 

Image before :- 

![Before](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/before.png)
![After](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/after.png)


## Architecure of Model. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GreyScale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|						28x28x48						|
| Max pooling	      	| 2x2 stride, valid padding , outputs 14x14x48		|
| Convolution 3x3	    | 1x1 Stride , valid padding , outputs 10x10x96 					|
| RELU	  | 10X10X96        									|
| Max pooling			|  2x2 stride , valid padding , output 5x5x96        				|
| Convolution 3x3	    | 2x2 Stride , valid padding , outputs 3x3x172					|
| RELU	  | 3x3x172       									|
| Max pooling			|  2x2 stride , valid padding , outputs 2x2x172        				|
|	Flatten					|		outputs 688										|
|	Fully Connected - 1 					|		outputs 84									|
|	Fully Connected - 2 					|		outputs 43									|

To train the model, 30 epochs of batch size 128 with learning rate 0.01 with the use of adam optimizer and softmax cross entropy with logits.

![Architecture of LeNet 1 layer](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png)


## Results 

My final model results were:
* validation set accuracy of 97.1 
* test set accuracy of 95.4
 

## Test a Model on New Images

Here are few traffic signs found on the web:

![Priority Road](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/new_signs/12_priority_road.jpg) 
![Yield](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/new_signs/13_yield.jpg)
![Stop](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/new_signs/14_stop.jpg) 
![Unknown  to the model](https://raw.githubusercontent.com/Shreyas3108/Traffic-Sign-Classifier/master/new_signs/99_unknown_sign.jpg)
The last image might be difficult to classify because it's not known in the dataset. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road     		| Priority road   									| 
| No-Entry     			| No-Entry										|
| Yield					| Yield											|
| Stop	      		| Stop					 				|
| Keep right	      		| Keep right					 				|
| Parking			| Stop      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%

For the first image, the model is relatively sure that this is a priority road sign (probability of 1.0), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road									| 
| 0.0     				| Traffic Signal 										|
| 0.0					| Bicycle crossing				|
| 0.0	      			| No vehicles				 				|
| 0.0				    | Roundabout mandatory      							|

## Conclusion 

The model performs very well on the basis of the test set as well as signs from the internet. With furthermore training set or more data , the model would be able to evaluate the images accurately. Furthermore processing of images such as rotation can be done. 
