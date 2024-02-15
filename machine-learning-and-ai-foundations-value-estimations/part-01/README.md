# Introduction to Value Estimations

Let's create a basic program to predict a house's value using `simple_value_estimator.ipynb`. The function, `estimate_house_value`, estimates a house's price based on two attributes: its size in **square feet** and the **number of bedrooms**. 

Initially, we assume every house is worth at least $50,000. To calculate the house's value, we add $92 for each square foot and $10,000 for each bedroom. This way, the function returns the house's estimated value considering its size and bedrooms.

# Linear Regression

Linear regression is a method where we model the value of a target based on one or more input features.

> We use a set of parameters, known as weights, to determine how each feature influences the target value. In the simplest form of machine learning, linear regression is a fundamental algorithm where the goal is to find the best set of weights that accurately predict the target value.

Unlike traditional programming where these weights are manually specified, in machine learning, the computer automatically identifies the optimal weights by analyzing the training data. This process allows the model to learn the relationship between the input features and the target value, enabling it to make predictions on new, unseen data.

In the context of linear regression, the cost function plays a crucial role in determining the optimal set of weights for the model. The cost function, often referred to as the "loss function" or "error function," measures the difference between the actual values in the training data and the values predicted by our model. Essentially, it quantifies how "wrong" or "off" our model's predictions are.

## Introduction to the Cost Function
When building machine learning models for tasks like estimating house prices, we rely on a metric known as the cost function to measure how far off our predictions are. This function calculates the total error between the predicted values and the actual values in our dataset. By averaging the squared differences (errors) across all data points and dividing by the number of houses, we obtain a **mean squared error** that represents the average error per house.

```markdown

Mean Squared Error Calculation

To calculate the mean squared error (MSE), which represents the average error per house in our dataset, we follow these steps:
1. Compute the squared difference between the predicted values and the actual values for each data point in the dataset.
2. Sum up all these squared differences.
3. Divide the total by the number of houses (data points) in the dataset to get the average.

This process gives us the mean squared error (MSE), a standard metric for evaluating the performance of regression models.

```


```python
def calculate_mse(actual_values, predicted_values):
    """
    Calculate the mean squared error between actual and predicted values.
    
    Parameters:
    - actual_values: A list of actual values.
    - predicted_values: A list of predicted values.
    
    Returns:
    - The mean squared error (MSE) as a float.
    """
    differences = [actual - predicted for actual, predicted in zip(actual_values, predicted_values)]
    squared_differences = [diff ** 2 for diff in differences]
    mse = sum(squared_differences) / len(actual_values)
    return mse

# Example usage
actual_values = [100000, 150000, 120000]
predicted_values = [95000, 145000, 125000]
mse = calculate_mse(actual_values, predicted_values)
print(f"Mean Squared Error: {mse}")
Mean Squared Error: 25000000.0
```


### The Role of the Cost Function
The cost function serves as a critical indicator of the performance of our model, quantifying the "cost" of using certain weights (parameters) in our predictions. The ultimate objective in machine learning model training is to minimize this cost, thereby enhancing the model's accuracy. A cost function value of zero signifies perfect predictions, but as the cost increases, so does the discrepancy between our model's estimates and the actual values.

### Generalizing the Cost Function
We express the cost function in a more general form by summing the squared differences between each predicted value and its corresponding actual value, then dividing by the total number of data points. This formulation allows us to systematically evaluate the impact of different weights on the model's performance.

### Optimizing with Gradient Descent
To find the set of weights that minimizes the cost, we employ optimization algorithms, with gradient descent being a prevalent choice. Gradient descent iteratively adjusts the weights in small increments in a direction that reduces the cost. Starting with random weights, it tweaks them thousands of times until the cost function reaches its minimum possible value or until further adjustments fail to significantly reduce the cost.

### Implementing Gradient Descent
The process begins by inputting our initial cost function and random weights into the gradient descent algorithm. Through iterative refinement, gradient descent converges on the optimal weights that minimize the cost function. Once the optimization process is complete, the algorithm returns these optimal weights, which can then be used to make accurate predictions.

## Recap and Conclusion
To summarize, the journey to finding the best weights for a simple home value estimator involves: modeling the problem with an equation, quantifying errors with a cost function, and employing gradient descent to minimize these errors. While modern machine learning libraries automate these steps, including running gradient descent, understanding the underlying principles is crucial for solving complex problems and improving model performance.
