import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from scipy.ndimage import interpolation
from scipy import misc
from cv2 import *
from load_data import load_mnist
from nn_utils import *
from spatialtransformer import *
import sys
def rotation(imgs):
    # rotates individual images
    img_np = imgs.get_value()
    batch_size, img_size = img_np.shape
    output = np.zeros((batch_size, 28, 28))
    for i in range(batch_size):
        val = np.random.uniform(low=90.0*-1, high=90.0)
        output[i] = interpolation.rotate(img_np[i].reshape(28,28), val, reshape=False)
    output = output.reshape(batch_size, img_size)
    output_shared = theano.shared(np.asarray(output, dtype = theano.config.floatX), borrow = True)
    return output_shared

def rts_combo(imgs):
    # rotate, translate, and scale individual images
    img_np = imgs.get_value()
    batch_size, img_size = img_np.shape
    output = np.zeros((batch_size, 28, 28))
    for i in range(batch_size):
        rotate = np.random.uniform(-45.0, 45.0)
        scale = np.random.uniform(0.7, 1.2)
        m = np.random.randint(-7,7, size = 2) # number of pixels to translate
        M = getRotationMatrix2D((28/2,28/2),rotate,1)
        dst = warpAffine(img_np[i].reshape(28,28), M, (28,28))
        M2 = np.float32([[1,0,m[0]],[0,1,m[1]]])
        output[i] = warpAffine(dst, M2, (28,28))
    output = output.reshape(batch_size, img_size)
    output_shared = theano.shared(np.asarray(output, dtype = theano.config.floatX), borrow = True)
    return output_shared

def proj_transform(imgs):
    img_np = imgs.get_value()
    batch_size, img_size = img_np.shape
    output = np.zeros((batch_size, 28, 28))
    for i in range(batch_size):
        scale = np.random.uniform(0.7, 1.0)
        dst = resize(img_np[i].reshape(28, 28), None, fx = scale, fy = scale)
        offset = 14 * scale
        pst1 = np.float32([[14 - offset, 14 - offset], [14 + offset, 14 - offset], [14 - offset, 14 + offset], [14 + offset, 14 + offset]])
        m = np.random.normal(0, 5, size = 4)
        pst2 = np.float32([[14 - offset + m[0], 14 - offset + m[0]],
                [14 + offset + m[1], 14 - offset + m[1]],
                [14 - offset + m[2], 14 + offset + m[2]],
                [14 + offset + m[3], 14 + offset + m[3]]])
        M = getPerspectiveTransform(pst1,pst2)
        output[i] = warpPerspective(dst,M,(28,28))
    output = output.reshape(batch_size, img_size)
    output_shared = theano.shared(np.asarray(output, dtype = theano.config.floatX), borrow = True)
    return output_shared

def identity(img):
    return img
def plot(img, dim=28):
    # plots a single image
    plt.imshow(np.reshape(img,(dim, dim)), cmap="gray")
    plt.xticks([]); plt.yticks([])
    plt.show()

def run(network_type='cnn', include_st=True, transformation = rotation, iterations=1.5e5, n_epochs=200, 
    batch_size=256, learning_rate=0.01, decay_param=0.1, activation=T.nnet.relu,
    verbose=True, runallmode = False):
    if runallmode == True:
        num_modes = 2
    else:
        num_modes = 1
    # load MNIST dataset from local directory 
    print('...loading the dataset')
    dataset = load_mnist()

    # partition into relevant datasets
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x , test_set_y  = dataset[2]
    train_set_x

    # Get transformed dataset
    print('...Applying {0} to dataset'.format(transformation.__name__))
    train_set_x = transformation(train_set_x)
    valid_set_x = transformation(valid_set_x)
    test_set_x = transformation(test_set_x)
    # compute minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # output dataset info
    if verbose:
        print('Current training data size is %i' %train_set_x.shape[0].eval())
        print('Current validation data size is %i' %valid_set_x.shape[0].eval())
        print('Current test data size is %i' %test_set_x.shape[0].eval())


    print('...building the model')
    rng = np.random.RandomState(23455)

    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    ############################
    # FULLY CONNECTED NETWORK  #
    ############################

    fcn_params = {
        'input_dim': (batch_size, 1, 28, 28),
        'h_units': 128,
        'h_layers': 2,
        'theta_dim': 6,
        'L1': 0.00,
        'L2': 0.0001,
    }

    if network_type in ['fcn', 'FCN', 'fully-connected']:
        for mode in range(num_modes):
            print("...training fcn with include_st = {0}".format(include_st))

            # check if spatial transformer should be included
            if include_st:
                st = STN_FCN(
                    input_dim = fcn_params['input_dim'],
                    input = x.reshape((batch_size, 1, 28, 28))
                )
                fcn_input = st.output
                fcn_input = fcn_input.reshape((batch_size, 28*28))
            else:
                fcn_input = x

            classifier = MultiLayerPerceptron(
                rng=rng,
                input=fcn_input,
                n_in=28*28,
                n_hidden=fcn_params['h_units'],
                n_out=10,
                n_hiddenLayers=fcn_params['h_layers'],
                act_function=activation
            )

            # cost to minimize during training
            cost = (classifier.negative_log_likelihood(y)
                + fcn_params['L1'] * classifier.L1
                + fcn_params['L2'] * classifier.L2_sqr
            )

            # testing
            test_model = theano.function(
                inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: test_set_x[index * batch_size:(index + 1) * batch_size],
                    y: test_set_y[index * batch_size:(index + 1) * batch_size]
                }
            )

            # validation
            validate_model = theano.function(
                inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                }
            )
            if include_st:
                classifier.params = classifier.params + st.params
            # compute graident of cost with respect to parameters
            gparams = [T.grad(cost, param) for param in classifier.params]

            # specify how to update parameter
            updates = [
                (param, param - learning_rate * gparam)
                for param, gparam in zip(classifier.params, gparams)
            ]

            # training
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

            print('...training')

            train_nn(train_model, validate_model, test_model,
                n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
            include_st = not include_st
            print("----------------------------------------------------------------------\n\n")

    ############################
    #  CONVOLUTIONAL NETWORK   #
    ############################
    
    cnn_params = {
        'input_dim': (batch_size, 1, 28, 28),
        'filter': 32
    }

    if network_type in ['cnn', 'CNN', 'convolutional']:
        for mode in range(num_modes):
            print("...training cnn with include_st = {0}".format(include_st))

            if include_st:
                print "...Apply STN before CNN"
                st = STN_CNN(
                    input_dim= cnn_params['input_dim'],
                    img=x.reshape((batch_size, 1, 28, 28)),
                    nconvs=[20,20],
                    downsampling=0.5,
                    scale=2
                )

                cnn_input = st.output
            else:
                cnn_input = x.reshape((batch_size, 1, 28, 28))

            # first convolutional pooling layer
            # filtering reduces to 20, maxpooling to 10
            # 4D output tensor is thus of shape (batch_size, 32, 10, 10)
            layer0 = ConvPoolLayer(
                rng=rng,
                input=cnn_input,
                image_shape=(batch_size, 1, 28, 28),
                filter_shape=(cnn_params['filter'], 1, 9, 9),
                poolsize=(2,2)
            )

            # second convolutional pooling layer
            # filter reduces to 4, maxpooling to 2
            # 4D output tensor is thus of shape (batch_size, 32, 2, 2)
            layer1 = ConvPoolLayer(
                rng=rng,
                input=layer0.output,
                image_shape=(batch_size, cnn_params['filter'], 10, 10),
                filter_shape=(cnn_params['filter'], cnn_params['filter'], 7, 7),
                poolsize=(2,2)
            )

            # classification
            layer2 = LogisticRegression(
                input=layer1.output.flatten(2),
                n_in=cnn_params['filter']*2*2,
                n_out=10
            )

            # cost we minimize during training
            cost = layer2.negative_log_likelihood(y)

            # testing
            test_model = theano.function(
                [index],
                layer2.errors(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

            # validation
            validate_model = theano.function(
                [index],
                layer2.errors(y),
                givens={
                    x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                    y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

            # list of model parameters to be fitted by gradient descent
            params = (layer2.params + layer1.params + layer0.params)
            if include_st:
                params += st.params

            # list of gradients for all model parameters
            grads = T.grad(cost,params)

            # specify how to update parameters
            updates = [
                (param_i, param_i - learning_rate * grad_i) 
                for param_i, grad_i in zip(params, grads)
            ]

            # training
            train_model = theano.function(
                [index],
                cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

            print('...training')

            train_nn(train_model, validate_model, test_model,
                n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
            include_st = not include_st
            print("----------------------------------------------------------------------\n\n")

if __name__ == '__main__':
    network = sys.argv[1]
    run(network_type = network)