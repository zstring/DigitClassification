import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import exp
from math import log

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    output = np.zeros((1,z.shape[1]))
    for i in range(z.shape[1]):
        output[0][i] = 1 / (1 + exp(-1 * z[0][i]))
    #your code here        
    return  output


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    path = '/home/inspire/Dropbox/UB/ML/Project/pa1/basecode/mnist_all.mat'
    mat = loadmat(path) #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    #Your code here
    validation_size = 1000

    train = mat.get('train0')
    train_data = np.zeros((1,train.shape[1]))
    train_label = np.zeros((1, 10))
    validation_data = np.zeros((1,train.shape[1]))
    validation_label = np.zeros((1,10))
    test_data = np.zeros((1,train.shape[1]))
    test_label = np.zeros((1,10))
    
    for i in range(10):
        train = mat.get('train' + str(i))
        inputSize = range(train.shape[0])
        inputSize = 5000
        randomIndex = np.random.permutation(inputSize)
        vData = train[randomIndex[0:validation_size],:]
        tData = train[randomIndex[validation_size:],:]
        validation_data = np.concatenate((validation_data,vData))
        true_label = np.zeros((validation_size,10))
        true_label[:,i] = 1;
        #validation_label = np.concatenate((validation_label, np.zeros((1000,1)) + i))
        validation_label = np.concatenate((validation_label, true_label))
        train_data = np.concatenate((train_data,tData))
        true_label = np.zeros((inputSize-validation_size, 10))
        true_label[:,i] = 1
        train_label = np.concatenate((train_label, true_label))
        #train_label = np.concatenate((train_label,np.zeros((train.shape[0]-1000, 1)) + i))
        #
        test = mat.get('test' + str(i))
        inputSizeTest = test.shape[0]
        #inputSizeTest = 5000
        test_data = np.concatenate((test_data,test[0:inputSizeTest,:]))
        true_label = np.zeros((inputSizeTest, 10))
        true_label[:,i] = 1
        test_label = np.concatenate((test_label, true_label))
        #test_label = np.concatenate((test_label, np.zeros((test.shape[0],1)) + i))
        
    validation_data = np.concatenate((validation_data,np.ones((validation_data.shape[0],1))),1)
    train_data = np.concatenate((train_data,np.ones((train_data.shape[0], 1))), 1)
    test_data = np.concatenate((test_data,np.ones((test_data.shape[0],1))),1)    
    return train_data/255, train_label, validation_data/255, validation_label, test_data/255, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    print 'Printing....'
    w1 = params[0:(n_hidden + 1) * (n_input + 1)].reshape((n_hidden+1, (n_input + 1)))
    w2 = params[((n_hidden+1) * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    #w1 = params[0:n_hidden * (n_input)].reshape( (n_hidden, (n_input)))
    #w2 = params[(n_hidden * (n_input)):].reshape((n_class, (n_hidden)))
    obj_val = 0  
    sizeWithBias = training_data.shape[1] + 1
    tData = np.zeros((training_data.shape[0],training_data.shape[1]+1))
    tData[:,:-1] = training_data;
    tData[0:,training_data.shape[1]-1:training_data.shape[1]] = 0
    tData[1:,training_data.shape[1]-1:training_data.shape[1]] = 1
    aj = np.zeros((1, n_hidden+1))
    #deltaL = np.zeros((1, 10))
    grad_w1 = np.zeros((n_hidden+1,sizeWithBias));
    grad_w2 = np.zeros((n_class, n_hidden+1))
    for i in range(n_input):
        for j in range(n_hidden+1):
            aj[0][j] = np.dot(w1[j], tData[i])
        zj = sigmoid(aj)
        bl = np.zeros((1, n_class))
        for l in range(n_class):
            bl[0][l] = np.dot(w2[l], zj[0])
        ol = sigmoid(bl)
        deltaL = (ol - training_label[i])
        ####################################################3
        #Objective Function
        # Sigma for all output yl(ln ol) + (1-yl) (ln (1-ol))        
        obj_val = obj_val + sum( sum (training_label[i] * np.log(ol) + 
                    (1 - training_label[i]) * np.log(1-ol)))
        ########################a############################3
        #Gradient descent for Hidden  Weights
        ## (1 * classNode) * ((classNode) * hiddenNodes) == 1 * hiddenNodes
        oneCrossW2 = np.dot(deltaL, w2) 
        ## (1 * hiddenNodes) * (1 * hiddenNodes) * (1 * hiddenNodes) = 1 * hiddenNodes
        oneCrossW2 = (1 - zj) * zj * oneCrossW2
        ## (1 * hiddenNodes).T * (1 * inputNodes) == hiddenNodes * inputNodes
        grad_w1 = grad_w1 + oneCrossW2.T * tData[i]
        
        #Gradient descent for Output Weights 
        # (1 * classNode).T * (1 * hiddenNodes) == classNodes * hiddenNodes       
        grad_w2 = grad_w2 + deltaL.T * zj
        ####################################################3        
    grad_w1 = grad_w1 / n_input
    grad_w1[:,n_input] = 0
    grad_w2 = grad_w2 / n_input
    grad_w2[:,n_hidden] = 0
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    obj_val = obj_val / n_input
    #Your code here
    #
    #
    #
    #
    #
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    n_hidden = w1.shape[0] - 1
    n_class = w2.shape[0]
    aj = np.zeros((1, n_hidden+1))
    
    tData = np.zeros((data.shape[0],data.shape[1]+1))
    tData[:,:-1] = data;
    tData[:,data.shape[1]-1:data.shape[1]] = 1
    data = tData;
    labels = np.zeros((data.shape[0],n_class))
    for i in range(data.shape[0]):
        for j in range(n_hidden):
            aj[0][j] = np.dot(w1[j], data[i])
        zj = sigmoid(aj)
        bl = np.zeros((1, n_class))
        for l in range(n_class):
            bl[0][l] = np.dot(w2[l], zj[0])
        ol = sigmoid(bl)
        maxIndex = ol.argmax(axis=1)
        ol[ol == ol[0,maxIndex]] = 1
        ol[ol < ol[0,maxIndex]] = 0
        labels[i] = ol
    #Your code here
    
    return labels
    
"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden+1);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 5}    # Preferred value.50

#nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
#w1 = nn_params.x[0:(n_hidden+1) * (n_input + 1)].reshape( (n_hidden+1, (n_input + 1)))
#w2 = nn_params.x[((n_hidden+1) * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
w1 = initial_w1
w2 = initial_w2
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print '\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%'

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print '\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%'


predicted_label = nnPredict(w1,w2,test_data)
print predicted_label.shape
print test_label.shape
#find the accuracy on Validation Dataset

print '\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%'