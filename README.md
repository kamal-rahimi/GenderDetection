# Gender recognition using Convolutional Neaural Networks (CNN)

## Model Description
The gender detection model consists of a face detection/croption step followed by Convolutional Neaural Networks classifier.

The face area in an input image is first detected using HAAR Classifier. The cropped face area is fed to Convolutional Neural Network which is composed of two pairs of convoution-max_poll-dropout layers followed by two fully connected layers. The outputs of the fully connected layers are fed to a softmax layer with two output to classify genders (Male or Female). The activation function of each layer are Exponential Linear Unit (ELU). 

Cross entrhopy is used to measure classification loss and model is trained using Adam optimizer.

The train data is 1356 face images from Colorfret dataset. The test data is another set of 339 face images Colorfret dataset. Both train and test data are balanced for gender using oversamling.

The model can predict gender in test data with accuracy of 96.7%. :)

The network stucture is depicted below: 

						      Input image
					      		   |
					      |-------------------------|
					      | face detection/croption |
					      |  using HAAR Classifier  |
					      |-------------------------|
							   |
						   Face image (64x64x1) 
							   |
						 |-------------------|
						 | conv2D (8,4x4,1)  | 
						 | ACT: ELU          |
						 |-------------------|
							   |
						  |------------------|
						  | max_poll (4x4,4) |
						  |------------------|
							   |
						    |-------------|
						    | dropuot 10% |
						    |-------------|
							   |
						 |-------------------|
						 | conv2D (32,4x4,1) | 
						 | ACT: ELU          |
						 |-------------------|
							   |
						  |------------------|
						  | max_poll (4x4,4) |
						  |------------------|
							   |
						    |-------------|
						    | dropuot 10% |
						    |-------------|
							   |
					       |------------------------|
					       | fully-connected (512)  |
					       | ACT: ELU               |
					       |------------------------|
							   |
					        |----------------------|
					        | fully-connected (2)  |
					        |----------------------|
							   |
						|----------------------|
						| 	softmax        |
						|----------------------|
						     |           |
						    Male       Female	

	



## How to use the model

### train_gender.py
Creats a model and trains it based on the Colorfret image dataset to detect gender of a person in an image.

Example usage:
```
$ python3 train_gender.py
```
Note a trained model based on 1356 face images from Colorfret dataset is icluded in this repository.


### predict.py
Predict the gender of the person in an input image.

Example usage:
```
$ python3 predict.py -p "./data/test/image.jpg"
```
The detected gender and model confidence (probabity of the predicted gender) will be shown in the image.

### predict_from_camera.py

Opens system camera and continuously detecs the gender of the person visble in the system camera.

Example usage:
```
$ python3 predict_from_camera.py 
```

