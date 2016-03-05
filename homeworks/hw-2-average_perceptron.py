import random, itertools
#random.seed(a=42)
import pdb

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

class Perceptron:
    'A simple Perceptron implementation.'
    def __init__(self, weights, bias, alpha=0.1):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
    
    def propagate(self, x):
        return self.activation(self.net(x)) 
        
    def activation(self, net):
        if net > 0:
            return 1
        return 0
    
    def net(self, x):
        return np.dot(self.weights, x) + self.bias
    
    def learn(self, x, y):
        y_hat = self.propagate(x)
        self.weights = [w_i + self.alpha*x_i*(y-y_hat) for (w_i, x_i) in zip(self.weights, x)]
        self.bias = self.bias + self.alpha*(y-y_hat)
        return np.abs(y_hat - y)


class AveragedPerceptron:
    ''' 
    Averaged perceptron variant.
    '''
    def __init__(self, weights, bias, alpha=0.1, weight_by_time=False):
        self.list_of_weights = [weights]
        self.list_of_biases  = [bias]
        # set the average weights/bias
        self.average_weights = np.mean(self.list_of_weights, axis=0)
        self.average_bias    = np.mean(self.list_of_biases)
        self.weight_by_time  = weight_by_time
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.time_weights = [1]
        self.t = 1
    
    def propagate(self, x, average=False):
        return self.activation(self.net(x, average=average)) 
        
    def activation(self, net):
        if net > 0:
            return 1
        return 0
    
    def net(self, x, average=False):
        if average:
            #import pdb; pdb.set_trace()
            return np.dot(self.average_weights, x) + self.average_bias
        return np.dot(self.weights, x) + self.bias

    def learn(self, x, y):
        y_hat = self.propagate(x)
        y_hat_avg = self.propagate(x, average=True)
        self.weights = [w_i + self.alpha*x_i*(y-y_hat) for (w_i, x_i) in zip(self.weights, x)]
        self.list_of_weights.append(np.array(self.weights))
        self.bias = self.bias + self.alpha*(y-y_hat)
        self.list_of_biases.append(self.bias)

        self.t += 1
        if self.weight_by_time:
            self.time_weights.append(self.t) # +1 to account for zero-indexing
        else:
            self.time_weights.append(self.t)

        #pdb.set_trace()

        # set the average weights/bias
        self.average_weights = np.average(self.list_of_weights, axis=0, weights=self.time_weights)
        self.average_bias    = np.average(self.list_of_biases, weights=self.time_weights)


        return np.abs(y_hat_avg - y)


def plot_perceptron_threshold(perceptron, ax):
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    
    x2s = [threshold(perceptron, x1) for x1 in xlim]
    ax.plot(xlim, x2s)
    
    ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)

def threshold(perceptron, x_1):
    return (-perceptron.weights[0] * x_1 - perceptron.bias) / perceptron.weights[1]

def plot_data(data, ax):
    data[data.y==1].plot(kind='scatter', 
                         x='$x_1$', y='$x_2$', 
                         color='Red', ax=ax)
    data[data.y==0].plot(kind='scatter', 
                         x='$x_1$', y='$x_2$', 
                         color='Gray', ax=ax)
    ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1)

def plot_all(perceptron, data, t, ax=None):
    if ax==None:
        fig = plt.figure(figsize=(5,4))
        ax = fig.gca()
    plot_data(data, ax)
    plot_perceptron_threshold(perceptron, ax)
    
    ax.set_title('$t='+str(t+1)+'$')

def learn_data(perceptron, data):
    'Returns the number of errors made.'
    count = 0 
    for i, row in data.iterrows():
        count += perceptron.learn(row[0:2], row[2])
        #perceptron.t += 1

    return count


def run(size=50, perceptron_type="vanilla"):
    data = pd.DataFrame(columns=('$x_1$', '$x_2$'),
                    data=np.random.uniform(size=(size,2)))
    
    def condition(x):
        return int(np.sum(x) > 1)

    data['y'] = data.apply(condition, axis=1)


    fig = plt.figure(figsize=(5,5))
    ax = fig.gca()
    plot_data(data, ax)

    if perceptron_type == "vanilla":
        perceptron = Perceptron([0.1,-0.1],0.1)
    else: 
        perceptron = AveragedPerceptron([0.1,-0.1],0.1)

    f, axarr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8,6))
    axs = list(itertools.chain.from_iterable(axarr))
    for t in range(6):
        plot_all(perceptron, data, t, ax=axs[t])
        learn_data(perceptron, data)
    f.tight_layout()


    f.savefig("learning-plot-%s.pdf" % perceptron_type)

if __name__ == '__main__':
    print("running vanilla ...")
    run()
    print("running averaged perceptron...")
    run(perceptron_type="average")

