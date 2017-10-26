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
LeNet Architecture was used as shown in the udacity classroom session ,I already used preprocessing using one_hot encoder as i usually use it as a good practice .I first trained a two layer Convolution of size 3x3 using CPU which was an exhaustive 1 hour process and gave 93.9 % in validation and test accuracy was horrible , hence i figured that some tweaks had to be done in order to successfully run the model as i suspected that another layer on Convolution might be required . After which i installed tensorflow-gpu from pip command (I usually use floydhub to train my models but since the file size was above 150mb floyhub was of no use) which then led to training data in 10-12 minutes approx.But after that validation accuracy increased to 99.3% and test accuracy was around 84% .After which i added one more layer on Convolution and used dropout which gave a better accuracy to the test set , somewhere around 94-95%. Then i again tried by removing dropout which gave me similar result. Hence i went forward with this model which is 3 layer convolution , but with two 5x5 and one 3x3 without dropout as my final solution. 
Weight of first layer is of shape (5 , 5 , 1 , 48)
Weight of second layer is of shape (3 , 3 , 48 , 96)
Weight of third layer is of shape (3 , 3 , 96 , 172)
EPOCH = 30 
BATCH SIZE = 128 
OPTIMIZER = adam
![Adam Optimizer](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/05/Comparison-of-Adam-to-Other-Optimization-Algorithms-Training-a-Multilayer-Perceptron.png)
learning rate = 0.001

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GreyScale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|						28x28x48						|
| Max pooling	      	| 2x2 stride, valid padding , outputs 14x14x48		|
| Convolution 5x5	    | 1x1 Stride , valid padding , outputs 10x10x96 					|
| RELU	  | 10X10X96        									|
| Max pooling			|  2x2 stride , valid padding , output 5x5x96        				|
| Convolution 3x3	    | 2x2 Stride , valid padding , outputs 3x3x172					|
| RELU	  | 3x3x172       									|
| Max pooling			|  2x2 stride , valid padding , outputs 2x2x172        				|
|	Flatten					|		outputs 688										|
|	Fully Connected - 1 					|		outputs 84									|
|	Fully Connected - 2 					|		outputs 43									|


![Architecture of LeNet 1 layer](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-4-59-29-pm.png)


## Results 

My final model results were:
* validation set accuracy of 99.6 
* test set accuracy of 95.115
 

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
