import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    data=no.random.normal(0,1,n)
    plt.hist(data,bins=10,color="black",edgecolor="red")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    return data


def uniform_histogram(n):
    data=np.random.uniform(0,10,n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10)")
    plt.show()

    return data


def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5)")
    plt.show()

    return data

# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    data = np.array(data)
    return np.sum(data) / len(data)


def sample_variance(data):
    data = np.array(data)
    n = len(data)
    
    mean = sample_mean(data)
    squared_diff = (data - mean) ** 2
    variance = np.sum(squared_diff) / (n - 1)
    
    return variance


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    data = np.array(data)
    data = np.sort(data)  
    
    n = len(data)
    min = data[0]
    max = data[-1]
    median = np.median(data)
    
    if n % 2 == 0:
        lower_half = data[:n//2]
        upper_half = data[n//2:]
    else:
        lower_half = data[:n//2]      
        upper_half = data[n//2 + 1:]   
    
    q1 = np.median(lower_half)
    q3 = np.median(upper_half)
    
    return (min, max, median, q1, q3)


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):

    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    
    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    
    return cov

# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):

    x = np.array(x)
    y = np.array(y)
    
    # Sample variances
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    
    # Sample covariance
    cov_xy = sample_covariance(x, y)
    
    # Build 2x2 covariance matrix
    cov_matrix = np.array([[var_x, cov_xy],
                           [cov_xy, var_y]])
    
    return cov_matrix

