import numpy as np

np.random.seed(3)

LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

# XOR truth table
x_train = [np.array([1.0, -1.0, -1.0]),
           np.array([1.0, -1.0,  1.0]),
           np.array([1.0,  1.0, -1.0]),
           np.array([1.0,  1.0,  1.0])]
y_train = [0.0, 1.0, 1.0, 0.0]

def neuron_w(input_count):
    weights = np.zeros(input_count + 1)
    # bias weight = 0
    for i in range(1, (input_count + 1)):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights

# neuron weights, output, and error for 3 neurons
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]

def show_learning():
    print('Current weights:')
    for i, w in enumerate(n_w):
        print('neuron ', i,
              ': w0 =', '%5.2f' % w[0],
              ', w1 =', '%5.2f' % w[1],
              ', w2 =', '%5.2f' % w[2])
    print('----------------')
    
def forward_pass(x):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], x))  # neuron 0
    n_y[1] = np.tanh(np.dot(n_w[1], x))  # neuron 1
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))
    
def backward_pass(y_truth):
    global n_error
    error_prime = -(y_truth - n_y[2])  # derivative of loss fn
    derivative = n_y[2] * (1.0 - n_y[2])  # neuron 2 derivative (logistic)
    n_error[2] = error_prime * derivative
    derivative = 1.0 - n_y[0]**2  # neuron 0 derivative (tanh)
    n_error[0] = n_w[2][1] * n_error[2] * derivative
    derivative = 1.0 - n_y[1]**2  # neuron 1 derivative (tanh)
    n_error[1] = n_w[2][2] * n_error[2] * derivative
    
def adjust_weights(x):
    global n_w
    n_w[0] -= (x * LEARNING_RATE * n_error[0])
    n_w[1] -= (x * LEARNING_RATE * n_error[1])
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])

# training loop
all_correct = False
while not all_correct:
    all_correct = True
    np.random.shuffle(index_list)  # randomize order of training samples
    # train on all samples
    for i in index_list:
        forward_pass(x_train[i])
        backward_pass(y_train[i])
        adjust_weights(x_train[i])
        show_learning()
    # check if network converged
    for i in range(len(x_train)):
        forward_pass(x_train[i])  # run sample thru network to get output
        print('x1 = ', '%4.1f' % x_train[i][1],
              'x2 = ', '%4.1f' % x_train[i][2],
              'y = ', '%.4f' % n_y[2])
        if (((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or
            ((y_train[i] >= 0.5) and (n_y[2] < 0.5))):
            all_correct = False
