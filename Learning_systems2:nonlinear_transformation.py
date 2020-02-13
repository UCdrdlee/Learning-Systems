import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def compute_g(target_f,randnm):
    ones = np.ones((randnm.shape[0],1))
    matrix_x = np.append(ones,randnm,1)
    pseudo_inverted_x = np.linalg.pinv(matrix_x)
    regression_w = np.array([np.dot(pseudo_inverted_x, target_f)])
    return regression_w, matrix_x

def compute_g2(target_f,randnm):
    ones = np.ones(randnm.shape[0])
    x1 = x[:,0]
    x2 = x[:,1]
    x1x2 = x1*x2
    x1x1 = x1*x1
    x2x2 = x2*x2
    d = {'var1':ones, 'var2':x1, 'var3':x2,'var4':x1x2, 'var5':x1x1, 'var6':x2x2}
    input_matrix = pd.DataFrame(data = d)
    pseudo_inverted_x = np.linalg.pinv(input_matrix)
    regression_w = np.array([np.dot(pseudo_inverted_x, target_f)])
    return regression_w, input_matrix

# Print answer for HW2 Q8
N = 1000
Ein = []
for i in np.arange(N):
    x = np.random.uniform(low = -1, high = 1, size = (1000,2)) #Generate 1000 random points from 2D
    f_raw = x[:,0]*x[:,0] + x[:,1]*x[:,1] - 0.6 #Compute noiseless target output
    f = np.sign(f_raw)
    true_f = f
    random_idx = random.sample(range(1,1000),100) #Generate 100 random numbers between 1 and 1000
    f[random_idx] = -f[random_idx] #Flip the signs of the elements in f with indices chosen above
    noisy_f = f
    w,x = compute_g(noisy_f,x) #Return weight vector based on noisy outputs and input vector x
    x_t = np.transpose(x)
    h_raw = np.dot(w, x_t)
    h = np.sign(h_raw)
    booli = noisy_f == h
    booli_int = booli.astype(int)
    Ein.append(np.sum(booli_int)/1000)
print("Avg in-sample error for 1000 runs is: ", np.mean(Ein))

# Print answer for HW2 Q9
N = 1000
# d = {'w1':0, 'w2':0, 'w3':0,'w4':0, 'w5':0, 'w6':0}
wsum = pd.DataFrame(np.zeros(6))
for i in np.arange(N):
    x = np.random.uniform(low = -1, high = 1, size = (1000,2))
    f_raw = x[:,0]*x[:,0] + x[:,1]*x[:,1] - 0.6
    f = np.sign(f_raw)
    true_f = f
    random_idx = random.sample(range(1,1000),100)
    f[random_idx] = -f[random_idx]
    noisy_f = f
    w,x = compute_g2(noisy_f,x)
    df_w = pd.DataFrame(w[0])
    wsum = wsum + df_w
print("Average of the six weights over 1000 runs is: ", wsum.divide(1000))

# Print answer for HW2 Q10
weight_vector = wsum.divide(1000) # the vector of the six weights computed in Q9
weight_vector_t = np.transpose(weight_vector)
Eouts = []
for i in np.arange(1000):
    x = np.random.uniform(low = -1, high = 1, size = (1000,2))
    ones = np.ones(x.shape[0])
    x1 = x[:,0]
    x2 = x[:,1]
    x1x2 = x1*x2
    x1x1 = x1*x1
    x2x2 = x2*x2
    d = {'var1':ones, 'var2':x1, 'var3':x2,'var4':x1x2, 'var5':x1x1, 'var6':x2x2}
    input_matrix = pd.DataFrame(data = d)
    input_matrix_t = np.transpose(input_matrix)
    f_raw = x1*x1 + x2*x2 - 0.6
    f = np.sign(f_raw)
    truef = f
    random_idx = random.sample(range(1,1000),100)
    f[random_idx] = -f[random_idx]
    noisyf = f
    h_output = np.sign(np.dot(weight_vector_t, input_matrix_t)) # classification output of our hypothesis
    h_vs_f = h_output == noisyf
    Eout = 1 - np.sum(h_vs_f.astype(int))/1000
    Eouts.append(Eout)
print("Avg. out-of-sample error over 1000 runs is: ", np.mean(Eouts))
