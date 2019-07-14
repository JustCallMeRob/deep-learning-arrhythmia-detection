# Arrhythmia detection using deep neural networks

## Project Description
This project focuses on the training of deep neural networks for the detection of arrhythmia from datasets 
containing parameters of [QRS complexes](https://en.wikipedia.org/wiki/QRS_complex) taken from electrocardiograms 
probes from various healthy and unhealthy individuals.
The dataset is called [Arrhythmia Data Set](https://bigml.com/dashboard/dataset/5c4c9b9500a1e5464c004994) and it was provided by the University of California
Machine Learning Repository.It contains 452 data points, each with 280 distinct features.
Each data point represents a single patient which has had measurements taken of their heart
activity with the use of an electrocardiogram (ECG) with 6 probes placed around the patients
chest. These probes are used to measure the QRS complex for the respective patient, which is
a name for the combination of three of the graphical deflections seen on a typical
electrocardiogram.

## Data Preprocessing
Before any training can be done, the dataset needs to be preprocessed. Some of the features
are stored in non numerical formal and will require conversion. Such as the gender which can
be binarily stored as 0 for females and 1 for males, and all the boolean values which are true
and false can be converted this way as well.
Some of the features have missing values, for these i have filled them with the average values
of their respective columns. It is very important to note that this is not the most optimal solution
and could introduce biases in the neural network. A more optimal solution would be to discard
any data points with missing values, hower due to the relatively small size of the dataset i have
not chosen this route.
Some of the features have values that do not vary from one datapoint to the next, these do not
contribute any useful information and are discarded. After discarding them we are left with 262
distinct features per datapoint from the original 280.
The last feature is the class of the datapoint representing the type of arrhythmia the patient has.
Originally the purpose of the dataset was train a model to classify the data points into 16 distinct
classes. This would be arrhythmia classification, hower due to the fact that this project focuses
solely on the detection of arrhythmia, all the healthy patients have been marked with “0” and the
rest that have some form of arrhythmia with “1”. So the problem was converted into a binary
classification problem.
The next step was to convert all the values to float except for the classes and separate the
features from the classes into their respective arrays. Once separated, they were then split into
training set and testing set, with 20% of the original data being allocated for testing.
Lastly, the features were normalized, using sklearn.preprocessing StandardScaler.
After all of this preprocessing we end up with a dataset that look like this.
![alt text](https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/dataset.PNG)

## Creating the neural network model
A feed forward deep neural network was implemented for the task at hand, 
it was created using Keras running on top of Tensorflow.
Since we are implementing a deep neural network we already know we need at least two hidden layers of neurons.
So a densely connected layer of neurons was added to the model with and input
size of 262, representing all the features in a datapoint. These were connected to a layer of
hidden neurons of some size, with weights that have been initialized in a uniform manner. Then
another densely connected layer of neurons have been added to the model to connect the fist
hidden layer to the another hidden layer, also with weights being initialized in a uniform manner.
Another densely connected layer has been added to represent the output of the network, this
has one node which will represent the probability of arrhythmia. The closer the value is to “0”,
the more likely the patient is healthy, the closer it is to “1” the more likely the patient has some
underlying heart problems.
The [Adagrad optimize]() was used as well as the [Binary Cross Entropy]() loss fucntion.

## Training the neural network
Note that the number of hidden neurons has not been specified so far, this is due to the fact that this project
also tests to see what comfiguration of neurons in the hidden layer is the most optimal for the task at hand. This is achieved
by training and testing various number of neural network comfigurations.
As an example, a number of neurons between 20 and 300 was tested, with step size of 20 neurons. This
resulted in training a total of 225 different combinations of neurons for the first and second
hidden layers. After which their accuracies and losses were plotted on a 3d graph, with the X
axis representing the number of neurons in the first hidden layer, the Y axis representing the
number of hidden neurons in the second layer and the Z axis representing either accuracy or loss.
The batch size was set to 5 with 5 epochs of training.
The following figures show the results of training.
<p float="left">
  <img src="https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/train_acc_big.png" width="500" />
  <img src="https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/train_loss_big.png" width="500" /> 
</p>
As can be observed in Fig1. the training accuracy increases exponentially with small numbers of
neurons and then plateaus as we reach the a number as high as the number of input neurons.
In the same fashion we can observe that the training losses decrease with an increase in hidden
layer neurons, which is what we are looking for.
The testing of these models can be seen in the next figure.
![alt text](https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/test_acc_big.png)!![alt text](https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/test_loss_big.png)!

The following is the generated log for the final model with 262 input neurons, 300 neurons in the
first hidden layer, 220 neurons in the second hidden layer and one output, batch size of 32,
epoch count 10.
![alt text](https://github.com/JustCallMeRob/deep-learning-arrhythmia-detection/blob/master/data.PNG)!
The final result for this run was a test accuracy of 74% with 99% training accuracy.
The final 4 lines in the log represent some random predictions from the test data, on the left side
having the real solution answers and the right being the model predictions.

## Conclusion
Although the model acted as intended it was unable to reach testing accuracies higher than
80% and there was no real way to improve this. I suspect this is due to the fact that the dataset
had a relatively small number of datapoints, with a much larger dataset the model could reach a
higher testing accuracy and even possibly be able to categorize the different types of
arrhythmias. It is also possible that the missing values in the dataset have inserted a bias in the
model.
