import numpy as np
import matplotlib.pyplot as plt

# Generate a random line in 2D plane
def rand_line():
    x = np.random.uniform(low = -1, high = 1, size = (2,2))
    # compute coefficient
    coeff = np.polyfit(x[0],x[1],1)
    # generate a polynomial with the coeff
    polynom = np.poly1d(coeff)
    # generate points in x-axis
    x_axis = np.linspace(-1,1)
    # compute y values for x
    y_axis = polynom(x_axis)
    return polynom

def get_h(w, randnm):
    # Initialize the weight vector with elements w0, w1, and w2
    ones = np.ones((randnm.shape[0],1))
    # Generate x vector
    x = np.append(ones,randnm,1)
    # Compute inner product of vector w and x
    inner = np.inner(w,x)
    # Get sign of the inner product and return h
    if np.all(inner== 0):
        proj = inner
    else:
        proj = np.where(inner>0, 1,-1)
    return proj

def target_output(polynom,randnm):
    # compute target output
    target = polynom(randnm[:,0]) < randnm[:,1]
    target_int = target.astype(int)
    target_int[target_int < 1] = -1
    return target_int

def update_w(polynom,rand_point,w):
    ones = np.ones((rand_point.shape[0],1))
    one_rand_point = np.append(ones,rand_point,1)
    output_f = target_output(polynom,rand_point)
    # Compare h and the target output and adjust the weight vector
    w = w + output_f[0]*one_rand_point
    return w

randnm = np.random.uniform(low = -1, high = 1, size = (5,2))
w = np.array([0,0,0])
h = get_h(w,randnm)

def num_iterations(polynom,n):
    # Generate n random points and test on these points
    x = np.random.uniform(low = -1, high = 1, size = (n,2))
    # initialize iterations and weight vector
    iterations = 0
    w = np.array([0,0,0])
    output_f = target_output(polynom,x)
    output_g = get_h(w,x)
    while np.any(output_f != output_g):
        iterations = iterations +1
        booli = output_f != output_g
        k = np.arange(x[booli].shape[0])
        random_index = np.random.choice(k,1)
        random_pt = x[booli][random_index]
        w = update_w(polynom,random_pt,w)
        output_g = get_h(w,x)[0]
    return iterations

def num_iterations2(polynom,n):
    # Generate n random points and test on these points
    x = np.random.uniform(low = -1, high = 1, size = (n,2))
    # initialize iterations and weight vector
    iterations = 0
    w = np.array([0,0,0])
    output_f = target_output(polynom,x)
    output_g = get_h(w,x)
    while np.any(output_f != output_g):
        iterations = iterations +1
        booli = output_f != output_g
        k = np.arange(x[booli].shape[0])
        random_index = np.random.choice(k,1)
        random_pt = x[booli][random_index]
        w = update_w(polynom,random_pt,w)
        output_g = get_h(w,x)[0]
    return w

def run_many(n,m):
    list_iterations = []
    for x in range(n):
        polynom = rand_line()
        iter = num_iterations(polynom,m)
        list_iterations.append(iter)
    return sum(list_iterations)/len(list_iterations)

# Run 1000 times for N=10 and compute the average. Gives answer to HW1 Q7
avg = run_many(1000,10)
print(avg)
# Run 1000 tims for N=100 and compute the average. Gives answer to HW1 Q9
avg = run_many(1000,100)
print(avg)

def disagreement_prob(n,m,out_sample):
    pr_disagreement = []
    for x in range(n):
        polynom = rand_line()
        w = num_iterations2(polynom, m)
        x = np.random.uniform(low = -1, high = 1, size = (out_sample,2))
        output_g = get_h(w, x)
        output_f = target_output(polynom,x)
        agree = output_g == output_f
        agree_int = agree.astype(int)
        num_correct = np.sum(agree_int,axis=1)
        pr_disagreement.append(sum(num_correct)/out_sample)
    print(sum(pr_disagreement)/len(pr_disagreement))
    return agree_int

# Run 1000 times with N=10. For each run, generate 10000 random out-of-sample points to
# estimate probability of correct assignment (-1 or 1). Then average over 1000 runs.
# 1-(answer) gives the best answer for HW1 Q8
agree = disagreement_prob(1000,10,10000)

# Run 1000 times with N=100. For each run, generate 10000 random out-of-sample points to
# estimate probability of correct assignment (-1 or 1). Then average over 1000 runs.
# 1-(answer) gives the best answer for HW1 Q10
agree = disagreement_prob(1000,100,10000)
