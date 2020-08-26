import pandas as pd
import autograd.numpy.random as npr
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import collections
import pickle
from tqdm import tqdm
import pylab
import autograd.numpy as np
# import numpy as np
import autograd.scipy.stats.norm as norm
from autograd.scipy.special import expit as sigmoid
from autograd import grad
from autograd.misc.optimizers import adam
import pickle
import scipy

def initiate_k_fold_input(input,target,k=10):
    k_data_set = []
    k_target_set = []
    l = len(input)
    divide = np.random.randint(k, size=l)
    for i in range(k):
        k_data_set.append([])
        k_target_set.append([])
    for i in range(len(divide)):
        k_data_set[divide[i]].append(input[i])
        k_target_set[divide[i]].append(target[i])
    return k_data_set, k_target_set








def divide_data_by_k(input,target,k=10):
    """use data without cross validation. validate data is 1/k of the whole data """
    training_set = []
    validation_set = []
    validation_target = []
    training_target = []
    l = len(input)
    divide = np.random.randint(k,size = l)
    for i in range(len(divide)):
        if divide[i] == (k-1):
            validation_set.append(input[i])
            validation_target.append(target[i])
        else:
            training_set.append(input[i])
            training_target.append(target[i])
    return np.array(training_set),np.array(validation_set), np.array(training_target), np.array(validation_target)







def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [[scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n)]      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix."""
    for W, b in params:
        # w1 = np.array(W)
        # i2 = np.array(inputs)
        # print(i2.shape)
        # print(w1.shape)
        # print(inputs[0])
        # print(W[0])
#        print("inputs are",inputs)
#        print("W are",W)

        # print()

        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = np.tanh(outputs)           # nonlinear transformation
    return outputs

def loss_function(params, inputs,target, l2_lambda):
    """L2 regularization"""
    y = neural_net_predict(params,inputs)
    l = 0.5 * ((y-target)**2).sum
    r = 0.5 * (params[0][0]**2).sum
    L_reg = l + l2_lambda*r

    return L_reg

def update_gradient(GD, params,lr):
    r = []
    for i in range(len(GD)):
        GD_pair = GD[i]
        Param_pair = params[i]
        r_pair = [Param_pair[0] - GD_pair[0]*lr,Param_pair[1] - GD_pair[1]*lr ]
        r.append(r_pair)
    return r

def correct_ratio(label,output):
    total = label.size
    correct = 0.
    for i in range(len(label)):
        t = label[i]
        y = output[i]
        for i in range(len(t)):
            if t[i]-0.5 < y[i] and y[i] < t[i]+0.5:
                correct+=1
    return correct/total


def round_result(output):
    r = []
    for row in output:
        new_row = []
        for number in row:
            new_row.append(int(round(number)))
        r.append(new_row)
    return r

def round_result_non_negative(output):
    r = []
    for row in output:
        new_row = []
        for number in row:
            x = int(round(number))
            if x > 0:
                new_row.append(x)
            else:
                new_row.append(0)
        r.append(new_row)
    return r

def neural_net_predict_k_fold(trained_params, test_data):
    outputs = []
    for param in trained_params:
        outputs.append(neural_net_predict(param, test_data))
    return outputs

def round_and_combine_predictions(outputs,t = 4):
    f = []
    for output in outputs:
        f.append(round_result_non_negative(output))
    f = np.array(f)
    x = np.zeros(f[0].shape)
    for item in f:
        x += item
    for row in x:
        for i in range(len(row)):
            if row[i] >= t:
                row[i] = 1
            else:
                row[i] = 0
    return x

# def pick_n_samples(n,training_set,training_target):
#     batch_set = []
#     batch_target = []
#     l = len(training_set)
#     for i in range(l):
#         if np.random.randint(0, l) <= n:
#             batch_set.append(training_set[i])
#             batch_target.append(training_target[i])
#     return batch_set, batch_target



def pick_n_samples(n,training_set,training_target):

    l = len(training_set)

    x = np.random.randint(0, high =l, size =n)
    batch_set = training_set[x]
    batch_target = training_target[x]

    return batch_set, batch_target




input_nn = np.load("nn_data/common_crawl_20k_input_vector.npy",allow_pickle=True)
label_nn = np.load("nn_data/common_crawl_20k_label_vector.npy",allow_pickle=True)
test_nn = np.load("nn_data/common_crawl_test_vector.npy",allow_pickle=True)

print("lennnn",len(test_nn))

training_set, validation_set, training_target, validation_target = divide_data_by_k(input_nn,label_nn,k = 10)
print(training_set)
print(len(training_set))

N = input_nn.shape[0] # number of input word data
input_d = input_nn.shape[1] # dimension for each input data
output_d = label_nn.shape[1] # dimension for each output data

#----------------Tunable Hyperparameters-------------------------------------------
k = 5
param_scale = 0.1
l2_lambda = 0.1
layer_sizes = [input_d , 500 ,output_d]
learning_rate = 0.1
iterations = 3000

#----------------Tunable Hyperparameters-------------------------------------------

init_params = init_net_params(param_scale, layer_sizes)


def L2_LOSS(params):
    """L2 regularization function for grad"""
    y = neural_net_predict(params,training_set)
    l = 0.5 * ((y-training_target)**2).sum() / y.size
    r = 0.5 * (params[0][0]**2).sum() / params[0][0].size
    L_reg = l + l2_lambda*r
    return L_reg


def train(params, iters = 3000, lr = 5e-3 ):
    params_cur = params
    training_ratios = []
    validation_ratios = []
    for i in range(iters):
        grad_l2 = grad(L2_LOSS)
        GD = grad_l2(params_cur)
        # print(GD, lr)
        params_cur = update_gradient(GD, params_cur,lr)
        train_output_cur = neural_net_predict(params_cur, training_set)
        training_correct_ratio = correct_ratio(training_target, train_output_cur)
        training_ratios.append(training_correct_ratio)

        validation_output_cur = neural_net_predict(params_cur, validation_set)
        validation_correct_ratio = correct_ratio(validation_target, validation_output_cur)
        validation_ratios.append(validation_correct_ratio)

        print("Iteration {} L2_loss {} training_correct_ratio {} validation_correct_ratiko {}".format(i, L2_LOSS(params_cur),training_correct_ratio,validation_correct_ratio))

        # print(GD)
    return params_cur



# uncomment lines below to train without k-fold validation. Only once with 10% as validation
# opt_param = train(init_params,lr =learning_rate)
# output = neural_net_predict(opt_param, input_nn)
# ratio = correct_ratio(label_nn,output)
   #   save
# dbfile = open('nn_data/round_result', 'wb')
# pickle.dump(rounded , dbfile)
# dbfile.close()

# load image
# dbfile = open('nn_data/round_result', 'rb')
# rr = pickle.load(dbfile)

# get test output
# output = neural_net_predict(opt_param, test_nn)
# rounded = round_result(output)






#-------------------K fold cross validation training--------------------------------------------------------------------


k_fold_data, k_fold_target = initiate_k_fold_input(input_nn,label_nn,k=k)
#
# def L2_LOSS(params):
#     """L2 regularization function for grad"""
#     y = neural_net_predict(params,training_set)
#     l = 0.5 * ((y-training_target)**2).sum() / y.size
#     r = 0.5 * (params[0][0]**2).sum() / params[0][0].size
#     L_reg = l + l2_lambda*r
#     return L_reg

def train_k_fold(k_data,k_target, iters = 1000, lr = 5e-3, k = 10):
    trained_param = []
    training_ratio = []
    validation_ratio = []
    for i in range(k):
        print("---------------------------------------k is ", i, " ---------------------------------------------------")
        validation_set = np.array(k_data[i])
        # print("valid set shape", validation_set.shape)
        validation_target = np.array(k_target[i])
        # print("valid target shape", validation_target.shape)
        training_set = []
        training_target = []
        for j in range(k):
            if j != i:
                training_set.extend(k_data[j])
                training_target.extend(k_target[j])
        training_set = np.array(training_set)
        training_target = np.array(training_target)
        # print("training set shape", training_set.shape)
        # print("training target shape", training_target.shape)

        # without stochastisity, define loss function here, use training_set instead of small_batch_set
        # def loss_function(params):
        #     """L2 regularization function for grad"""
        #     y = neural_net_predict(params,training_set)
        #     l = 0.5 * ((y-training_target)**2).sum() / y.size
        #     r = 0.5 * (params[0][0]**2).sum() / params[0][0].size
        #     L_reg = l + l2_lambda*r
        #     return L_reg


        params_cur = init_net_params(param_scale, layer_sizes)
        # training_ratios = []
        # validation_ratios = []
        for iter in range(iters):
            # ADD stochastisity, only use a batch with N samples from the whole set
            small_batch_set, small_batch_target = pick_n_samples(2000,training_set,training_target)
            small_batch_set = np.array(small_batch_set)
            small_batch_target = np.array(small_batch_target)

            def loss_function(params):
                """L2 regularization function for grad"""
                y = neural_net_predict(params, small_batch_set)
                l = 0.5 * ((y - small_batch_target) ** 2).sum() / y.size
                r = 0.5 * (params[0][0] ** 2).sum() / params[0][0].size
                L_reg = l + l2_lambda * r
                return L_reg
            grad_l2 = grad(loss_function)
            GD = grad_l2(params_cur)
            # print(GD, lr)
            params_cur = update_gradient(GD, params_cur, lr)
            train_output_cur = neural_net_predict(params_cur, training_set)
            training_correct_ratio = correct_ratio(training_target, train_output_cur)
            # training_ratios.append(training_correct_ratio)

            validation_output_cur = neural_net_predict(params_cur, validation_set)
            validation_correct_ratio = correct_ratio(validation_target, validation_output_cur)
            # validation_ratios.append(validation_correct_ratio)

            print("K {} Iteration {} L2_loss {} training_correct_ratio {} validation_correct_ratiko {}".format(i, iter, L2_LOSS(
                params_cur), training_correct_ratio, validation_correct_ratio))

        print("----------------------------------K fold = ", i, "is completed-----------------------------------------")
        trained_param.append(params_cur)

        train_output_cur = neural_net_predict(params_cur, training_set)
        training_correct_ratio = correct_ratio(training_target, train_output_cur)
        training_ratio.append(training_correct_ratio)

        validation_output_cur = neural_net_predict(params_cur, validation_set)
        validation_correct_ratio = correct_ratio(validation_target, validation_output_cur)
        validation_ratio.append(validation_correct_ratio)

    return  trained_param, training_ratio,validation_ratio

# comment the line below to not train in  k-fold
trained_param, training_ratio,validation_ratio = train_k_fold(k_fold_data, k_fold_target, iters = iterations, lr = learning_rate, k = k )


# dbfile = open('nn_data/trained_param_20k_k_10_i3000', 'wb')
# pickle.dump(trained_param , dbfile)
# dbfile.close()
# dbfile = open('nn_data/training_ratio_20k_k_10_i3000', 'wb')
# pickle.dump(training_ratio , dbfile)
# dbfile.close()
# dbfile = open('nn_data/validation_ratio_20k_k_10_i3000', 'wb')
# pickle.dump(validation_ratio , dbfile)
# dbfile.close()

# dbfile = open('nn_data/trained_param_k_10', 'rb')
# rr = pickle.load(dbfile)



# Get final result (K-FOLD)
# format = np.load("label_format.npy",allow_pickle=True)

# dbfile = open('nn_data/trained_param_k_10_i4000', 'rb')
# trained_param = pickle.load(dbfile)
# dbfile = open('nn_data/training_ratio_k_10_i4000', 'rb')
# trained_ratio = pickle.load(dbfile)
# dbfile = open('nn_data/validation_ratio_k_10_i4000', 'rb')
# validation_ratio = pickle.load(dbfile)

outputs = neural_net_predict_k_fold(trained_param, test_nn)
final_output = round_and_combine_predictions(outputs,t=3)

# transfer to list object to output csv file
L = []
for a in final_output:
    L.append(list(a))

dbfile = open('nn_data/final_output_20k_k_5_t_3', 'wb')
pickle.dump(L , dbfile)
dbfile.close()

# opt_param = train(init_params,lr =learning_rate)
# output = neural_net_predict(opt_param, input_nn)
# ratio = correct_ratio(label_nn,output)