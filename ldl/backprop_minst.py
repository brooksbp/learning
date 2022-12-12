import numpy as np
import matplotlib.pyplot as plt
import idx2numpy

np.random.seed(7)

LEARNING_RATE = 0.01
EPOCHS = 20


TRAIN_IMAGE_FILENAME = 'data/minst/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = 'data/minst/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME = 'data/minst/t10k-images-idx3-ubyte'
TEST_LABEL_FILENAME = 'data/minst/t10k-labels-idx1-ubyte'

def read_minst():
    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)
    
    # reshape (60000, 28, 28) to (60000, 784) to simplify feeding
    # input data to the network
    x_train = train_images.reshape((60000, 784))
    x_test = test_images.reshape((10000, 784))
    # standardize the data by taking (0-255) range values and scale
    # them down and center them around zero.
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev
    
    # one-hot encode outputs
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
        
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = read_minst()

index_list = list(range(len(x_train)))  # used to randomize samples

def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for i in range(neuron_count):
        for j in range(1, (input_count + 1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights

hidden_layer_w = layer_w(25, 784)
hidden_layer_y = np.zeros(25)
hidden_layer_error = np.zeros(25)

output_layer_w = layer_w(10, 25)
output_layer_y = np.zeros(10)
output_layer_error = np.zeros(10)

chart_x = []
chart_y_train = []
chart_y_test = []

def show_learning(epoch_no, train_acc, test_acc):
    global chart_x
    global chart_y_train
    global chart_y_test
    print('epoch no:', epoch_no,
          ', train_acc: ', '%6.4f' % train_acc,
          ', test_acc: ', '%6.4f' % test_acc)
    chart_x.append(epoch_no + 1)
    chart_y_train.append(1.0 - train_acc)
    chart_y_test.append(1.0 - test_acc)

def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()
    
def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    # activation fn for hidden layer
    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)
    output_layer_inputs = np.concatenate((np.array([1.0]), hidden_layer_y))
    # activation fn for output layer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, output_layer_inputs)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error
    # backprop error for each output neuron
    for i, y in enumerate(output_layer_y):
        error_prime = -(y_truth[i] - y)  # loss derivative
        derivative = y * (1.0 - y)  # logistic derivative
        output_layer_error[i] = error_prime * derivative
    for i, y in enumerate(hidden_layer_y):
        # get all the weights of the neurons that this one feeds to
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i + 1])
        error_weights_array = np.array(error_weights)
        # backprop error for this neuron
        derivative = 1.0 - y**2  # tanh derivative
        weighted_error = np.dot(error_weights_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w
    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LEARNING_RATE * error)
    output_layer_inputs = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= (output_layer_inputs * LEARNING_RATE * error)

# training loop
for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0
    for j in index_list:
        x = np.concatenate((np.array([1.0]), x_train[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training_results += 1
        backward_pass(y_train[j])
        adjust_weights(x)
        
    correct_test_results = 0
    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_test[j].argmax():
            correct_test_results += 1
            
    show_learning(i,
                  correct_training_results / len(x_train),
                  correct_test_results / len(x_test))

plot_learning()
