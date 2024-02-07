## Linear Regression implementation in Python from scratch.
Linear Regression stands as one of the simplest supervised machine learning algorithms, aiming to approximate the likelihood of a dependent variable converging based on one or multiple independent variables or features, through a linear equation. This equation is typically represented as:

$y = mx + b$ <br>
In this representation, $X$ represents a feature, and for multiple variables, the set of $X$ can be denoted as $X = {x1, x2, … xn}$, or in a vectorized form as $X = [x1, x2, …, xn]$, with m representing the weight of the corresponding feature within the feature vector $X$, and $b$ denoting the bias. By denoting the weight vector as w, the function can be expressed as:

$𝑓(𝑥) = 𝑤𝑥 + 𝑏,$

where 𝑥 is the feature vector with respect to the weight and bias w and b. The objective is to optimize the values of the weight and bias (w, b) such that for any prediction y, it closely fits the linear equation.

## Cost Function and Gradient Descent
To determine the distance between points $y$ and the line represented by the linear equation, a cost function $J(w, b)$ is defined as follows:

$𝑐𝑜𝑠𝑡(𝑖)=(𝑓_𝑤𝑏−𝑦(𝑖))² $

$𝐽(𝐰,𝑏)= 1/2𝑚 ∑ 𝑖=0, 𝑚−1 𝑐𝑜𝑠𝑡(𝑖) $

where $m$ is the number of samples.

Intuitively, the cost function can be optimized through a gradient descent mechanism. This involves finding the first-order derivatives of the cost function with respect to smaller steps, denoted as alpha.


## Python Implementation for single variable problem
Load the dataset using the following python code snippet: <br>

```python 
def load_data():
    data = np.loadtxt("data_file", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y
x_train, y_train = load_data()
```
Check the data in the train and test set: <br>
```python
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 

print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
```

The cost function is defined as:
```python
def compute_cost(x, y, w, b): 
    
    # number of training examples
    m = x.shape[0] 
    
    total_cost = 0

    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        total_cost = total_cost + (f_wb_i - y[i])**2
    total_cost = total_cost / (2 * m)

    return total_cost

```
Corresponding gradient descent to get the updated value of weight $w$ and bias $b$
```python

def compute_gradient(x, y, w, b): 
    
    # Number of training examples
    m = x.shape[0]
   
    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):                             
        error = (np.dot(x[i], w) + b) - y[i]                        
        dj_dw= dj_dw + error * x[i]
        dj_db = dj_db + error                       
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m
        
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = compute_gradient(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)
        
    return w, b, J_history, w_history
```
Finaly the implementation of the linear equation for all samples to make prediction is:

```python
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b
```
## Observation
After applying linear regression to the provided dataset, it is evident that by approximating the weight and bias (w, b) through the cost function and gradient descent, the equation satisfactorily fits the data without exhibiting signs of overfitting or underfitting. This implementation utilized 1500 iterations and a step count of 0.1 for gradient descent. Despite the naivety of the approach and a modest sample size of 97, the model’s performance is expected to improve when trained on larger datasets with more feature variables, enabling a more realistic prediction of profits. However, irrespective of dataset size, the underlying mathematical intuition remains unchanged.



A medium blog post is availabe for explaing this repository can be found [here](https://hasan-shahriar.medium.com/linear-regression-implementation-in-python-from-scratch-ced6ba267460)
