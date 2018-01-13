# Artificial-Neural-Network

This project analyses a fake bank dataset. called [Churn_Modelling.csv](Artificial-Neural-Network/Churn_Modelling.csv). 
This dataset has a large sample of this fake bank's customers. The dataset includes customer id, credit score, 
gender, age, tenure, balance, activeness, credit card, etc. The goal of this Artificial Neural Network (ANN) is 
to predict based on the dataset given, if a customer will leave the bank or stay (the probability). The ANN will be trained with 
Stochastic Gradient Descent.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and 
testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing Prerequisites

Libraries needed to be installed to run this project (For Windows and Linux):

1. [Tensorflow](https://www.tensorflow.org/) library , open source numerical computation library very efficient for fast numerical computation can run in your CPU and GPU.
2. [Theano](http://deeplearning.net/software/theano/) library , also open source numerical computation library very efficient for fast numerical computation can run in your CPU and GPU.
3. [Keras](http://deeplearning.net/software/theano/) library , wraps both Theano and Tensorflow libraries together.

[Installing Tensorflow tutorial](https://www.youtube.com/watch?v=gWfVwnOyG78) by Codacus on youtube, this 
tutorial uses anaconda navigator to create a new python environment and it is the same process to install Theano and Keras libraries. 

To make sure that tensorflow has been installed correctly try the following:
```python
import tensorflow as tf

hello = tf.constant('Hellow, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

## Fitting the ANN to the Training set

Compiling the [ann.py](link) file will train the ANN and give the following results

When using

```python
batch_size= 10
epochs= 100
```

When we apply Stochastic Gradient Descent to the ANN and fit to the training set the results are the following:

```
  10/8000 [..............................] - ETA: 0s - loss: 0.4617 - acc: 0.7000
 650/8000 [=>............................] - ETA: 0s - loss: 0.3838 - acc: 0.8446
1300/8000 [===>..........................] - ETA: 0s - loss: 0.4007 - acc: 0.8377
1930/8000 [======>.......................] - ETA: 0s - loss: 0.4088 - acc: 0.8311
2550/8000 [========>.....................] - ETA: 0s - loss: 0.3943 - acc: 0.8392
3180/8000 [==========>...................] - ETA: 0s - loss: 0.3917 - acc: 0.8403
3820/8000 [=============>................] - ETA: 0s - loss: 0.3960 - acc: 0.8361
4460/8000 [===============>..............] - ETA: 0s - loss: 0.3989 - acc: 0.8357
5090/8000 [==================>...........] - ETA: 0s - loss: 0.4021 - acc: 0.8334
5710/8000 [====================>.........] - ETA: 0s - loss: 0.4040 - acc: 0.8343
6360/8000 [======================>.......] - ETA: 0s - loss: 0.4017 - acc: 0.8352
6990/8000 [=========================>....] - ETA: 0s - loss: 0.4019 - acc: 0.8349
7630/8000 [===========================>..] - ETA: 0s - loss: 0.3994 - acc: 0.8351
8000/8000 [==============================] - 1s 79us/step - loss: 0.3993 - acc: 0.8345
Epoch 100/100
```

As you can see at Epoch 100 we get an accuracy of __83.45%__

### Predicting the Test set results

When the confusion matrix was made these were the results:

![alt text](..//cm.png "Confusion Matrix")

From the confusion matrix above, out of 2000 new observations we have:
```
1544 + 140 = 1684 correct predictions
265 + 51 = 316 incorect predictions
```

To get the accuracy we do:
```
1544 + 140 / 2000 = 0.842
```

This means that in new observations, where we did not train the ANN we get an accuracy of __84.2%!!__

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* Codacus for the tensorflow enviroment install tutorial