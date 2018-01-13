# Artificial-Neural-Network

This project analyses a fake bank dataset. This data set 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing Prerequisites

Libraries needed to be installed to run this project (For Windows and Linux):

1. [Tensorflow](https://www.tensorflow.org/) library , open source numerical computation library very efficient for fast numerical computation can run in your CPU and GPU.
2. [Theano](http://deeplearning.net/software/theano/) library , also open source numerical computation library very efficient for fast numerical computation can run in your CPU and GPU.
3. [Keras](http://deeplearning.net/software/theano/) library , wraps both Theano and Tensorflow libraries together.

[Installing Tensorflow tutorial](https://www.youtube.com/watch?v=gWfVwnOyG78) by Codacus on youtube, this tutorial uses anaconda navigator to create a new python environment and it is the same process to install Theano and Keras libraries. 

To make sure that tensorflow has been installed correctly try the following:
```python
import tensorflow as tf

hello = tf.constant('Hellow, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* Codacus for the tensorflow enviroment install tutorial