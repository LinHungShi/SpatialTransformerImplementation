import theano.tensor as T
from theano.tensor.nnet import conv
import theano
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import numpy as np
class SpatialTransformerLayer(object):

    def __init__(self, conv_input, theta, down_fraction = 1):    
        self.down_fraction = down_fraction
        self.output = affine_sampling(theta, conv_input, self.down_fraction)

def affine_sampling(theta, input, df):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))
    f_height = T.cast(height, 'float32')
    f_width = T.cast(width, 'float32')
    o_height = T.cast(f_height // df, 'int64')
    o_width = T.cast(f_width // df, 'int64')

    grid = create_meshgrid(o_height, o_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    Tg = T.dot(theta, grid)
    xs, ys = Tg[:, 0], Tg[:, 1]
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
  
    input_trans = bilinear_sampling(
        input_dim, xs_flat, ys_flat,
        df)
  
    output = T.reshape(input_trans,
                       (num_batch, o_height, o_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2) 
    return output

def create_linspace(start, stop, num):
    start = T.cast(start, 'float32')
    stop = T.cast(stop, 'float32')
    num = T.cast(num, 'float32')
    step = (stop-start)/(num-1)
    return T.arange(num, dtype='float32')*step+start
def create_meshgrid(height, width):

    xt = T.dot(T.ones((height, 1)),
               create_linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    yt = T.dot(create_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                 T.ones((1, width)))
   
    xt_flat = xt.reshape((1, -1))
    yt_flat = yt.reshape((1, -1))
    ones = T.ones_like(xt_flat)
    grid = T.concatenate([xt_flat, yt_flat, ones], axis=0)
    return grid

def rept(x, n_rep):
    rep = T.ones((n_rep,), dtype='int32').dimshuffle('x', 0)
    x = T.dot(x.reshape((-1, 1)), rep)
    return x.flatten()

def binlinear_sampling(img, x, y, df):
    # constants
   
    num_batch, height, width, channels = img.shape
    f_height = T.cast(height, 'float32')
    f_width = T.cast(width, 'float32')
    o_height = T.cast(f_height // downsample_factor, 'int64')
    o_width = T.cast(f_width // downsample_factor, 'int64')
    zero = T.zeros([], dtype='int64')
    y_max = T.cast(img.shape[1] - 1, 'int64')
    x_max = T.cast(img.shape[2] - 1, 'int64')
    o_x = (x + 1.0)*(f_width) / 2.0
    o_y = (y + 1.0)*(f_height) / 2.0
  
    x0 = T.cast(T.floor(o_x), 'int64')
    x1 = x0 + 1
    y0 = T.cast(T.floor(o_y), 'int64')
    y1 = y0 + 1

    x_floor = T.clip(x0, zero, x_max)
    x_ceil = T.clip(x1, zero, x_max)
    y_floor = T.clip(y0, zero, y_max)
    y_ceil = T.clip(y1, zero, y_max)
    dim1 = width*height
    dim2 = width
    base = rept(
        T.arange(num_batch, dtype='int32')*dim1, o_height*o_width)
    base_y_floor = base + y_floor*dim2
    base_y_ceil = base + y_ceil*dim2
    idxa = base_y_floor + x_floor
    idxb = base_y_ceil + x_floor
    idxc = base_y_floor + x_ceil
    idxd = base_y_ceil + x_ceil


    img_flat = img.reshape((-1, channels))
    I_a = img_flat[idxa]
    I_b = img_flat[idxb]
    I_c = img_flat[idxc]
    I_d = img_flat[idxd]

    # and finanly calculate interpolated values
    xf_f = T.cast(x_floor, 'float32')
    xc_f = T.cast(x_ceil, 'float32')
    yf_f = T.cast(y_floor, 'float32')
    yc_f = T.cast(y_ceil, 'float32')
    w_a = ((xc_f-x) * (yc_f-y)).dimshuffle(0, 'x')
    w_b = ((xc_f-x) * (y-yf_f)).dimshuffle(0, 'x')
    w_c = ((x-xf_f) * (yc_f-y)).dimshuffle(0, 'x')
    w_d = ((x-xf_f) * (y-yf_f)).dimshuffle(0, 'x')
    output = T.sum([w_a*I_a, w_b*I_b, w_c*I_c, w_d*I_d], axis=0)
    return output

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
        )
        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input   
class STN_CNN(object):
    def __init__(self, input_dim, img, nconvs = [20,20], downsampling = 1, scale = 1):
        input = img
        scale = int(scale)
        if scale != 1:
            input = T.signal.pool.pool_2d(img, ds = (scale,scale), ignore_border = True)
        rng = np.random.RandomState(12345)
        batch_size, channels, height, width = input_dim
        height /= scale
        width /= scale

        layer0 = ConvPoolLayer(
                rng,
                input=input,
                image_shape=(batch_size, channels, height, width),
                filter_shape=(nconvs[0], channels, 5, 5),
                poolsize=(2, 2)
            )
        layer1 = ConvPoolLayer(
                rng,
                input=layer0.output,
                image_shape=(batch_size, nconvs[0], (height - 4) / 2, (width - 4) / 2),
                filter_shape=(nconvs[1], nconvs[0], 5, 5),
                poolsize=(1, 1)
            )
        
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in= nconvs[1] * ((height - 4) / 2 - 4) * ((width - 4) / 2 - 4),
            n_out=20,
            activation = T.nnet.relu
        )
        print "Init localization to identity"
        W_values = np.zeros((20, 6), dtype = theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = np.array([1,0,0,0,1,0],dtype = theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in= 20,
            n_out=6,
            activation = None,
            W = W,
            b = b
        )
        theta = layer3.output
        theta = T.reshape(theta, (-1,6))
        self.input = input
        self.params = layer0.params + layer1.params + layer2.params + layer3.params
        self.output = SpatialTransformerLayer(input, theta, down_fraction = downsampling).output
        self.theta = theta
class STN_FCN(object):
    def __init__(self, input_dim, input, nhids = [32,32,32], downsampling = 1, activation = T.nnet.relu):
        batch_size, channel, height, width = input_dim
        img_size = height * width
        inputX = input.reshape((batch_size, channel, img_size))
        rng = np.random.RandomState(12345)
        layer0 = HiddenLayer(
            rng,
            input=inputX,
            n_in= img_size,
            n_out=nhids[0],
            activation = activation
        )
        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in= nhids[0],
            n_out= nhids[1],
            activation =  activation
        )
        layer2 = HiddenLayer(
            rng,
            input=layer1.output,
            n_in= nhids[1],
            n_out= nhids[2],
            activation =  activation
        )
        print "...Init localization to identity"
        
        W_values = np.zeros((nhids[1], 6), dtype = theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = np.array([1,0,0,0,1,0],dtype = theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in= nhids[2],
            n_out= 6,
            activation = None,
            W = W, 
            b = b
        )
        
        theta = layer3.output
        theta = T.reshape(theta, (-1,6))
        self.params = layer0.params + layer1.params + layer2.params + layer3.params
        self.output = SpatialTransformerLayer(input, theta, down_fraction = downsampling).output
        self.theta = theta
        self.input = input