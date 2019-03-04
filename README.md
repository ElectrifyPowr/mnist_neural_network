# Classifier of Handwritten Number Images
## Using a Deep Neural Network & the MNIST Dataset

This is an implementation of a deep neural network for <b>classifying handwritten numbers between 0 and 9</b>.<br>
<br><br>
## Steps
<br>
In order to execute this python script, please follow the following steps:
1. Create a folder called <b>mnist_dataset</b> in the same directory as the <b>neural_network.py</b> file
2. Download both training and test dataset as csv files from [here](https://pjreddie.com/projects/mnist-in-csv/)
3. Put both files into the <b>mnist_dataset directory</b> (from <i>step 1</i>) as <b>mnist_train.csv</b> and <b>mnist_test.csv</b>
4. Execute the <b>generate_rotated_images.py</b> script, which will generate two more files under the <i>mnist_dataset</i> directory called <b>rotated_mnist_train.csv</b> and <b>rotated_mnist_test.csv</b> (containing for each original image 2 more images rotated to the left & right by 10 degrees)
5. Change the hyperparameters as you like in the <b>neural_network.py</b> file
6. Train & test the neural network with ```python neural_network.py```


