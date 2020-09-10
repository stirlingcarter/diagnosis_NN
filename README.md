# diagnosis_NN

A neural network built with TensorFlow which utilizes a medical dataset to diagnose breast cancer with a success rate of 97%. 

## The data

We start with a completely public dataset from the University of Wisconsin, which consists of thousands of samples of some 30 key metrics. These metrics are diverse, but they all manage to quantify a breast tumor in some way - by measuring density, composition, placement, etc. The exact meaning of these measurements doesn't really matter to us. The important part is that each row of measurements has a key indicator, a 1 or a 0, to signal whether the measurements came from a benign tumor or a cancerous one.

## The goal

The goal is to develop a model that has the ability to decide on its own whether a row of metrics should end with a 1 or a 0. Our model should have seen so many rows in its lifetime that it knows whether a tumor is cancerouos or not just by looking at a few dozen metrics gathered from the patient. 

## Training

To initiate our model, we used our key metrics, or features, as input values to a neural network - our first layer. These discrete and continuous values would be summated, multiplied, or otherwise transformed and combined in subsequent layers. Our answer, a 1 or 0, or perhaps a negative or positive number, is simply some combination of the final layer. 


To begin, our model had a 40-60% success rate, which is useless. However, we can exploit simple rules of calculus to improve our success rates. After all, our model is nothing more than a polynomial! If we assign coefficients to our key summations or multiplications, we are in effect adding levers to our polynomial. We can use gradient descent to measure the direction our polynomial trends towards upon changing any given coefficient. The process is simple: iterate through each row of training data, and iterate through each coefficient in our polynomial - stop each time to measure the gradient descent, and take a small step (modify the coefficient) in the direction towards our desired outcome for that row (1, or 0? benign or cancerous?). Technically, we are minimizing an error function. If we do this enough times, we may just teach our model to identify cancer - we can set aside some training data as testing data to vet our model.


## Obstacles

There are a few obstacles we had to reconcile in training our model. One was the issue of local minima. If your instructions are to take a step towards the lowest point, you will never walk up and out of a valley, even if there is a deeper valley somewhere nearby. This can be mitigated by starting with many random values for our neural network, and choosing the most successful run as our model. 


Another problem is overfitting. A model is like saran wrap. We don't want to wrap tooo tightly around our training dataset - otherwise we run the risk of only marking a tumor cancerous if and only if it exactly matches a cancerous row of our dataset, or conversely marking a tumor benign if and only if it exactly matches a benign row of our dataset. To avoid this problem, we recognize diminishing returns in our iterative training algorithm and stop while we're ahead. We can also increase the number of layers, neurons, or modulate the type of arithmetic within our model. Trail and error is a crucial aspect of machine learning.



