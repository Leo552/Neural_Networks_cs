# Neural_Networks_cs
Neural networks in c# without the use of many third party libraries.

Features of the NN:
  - Mini batch gradient descent
  - Multi-threading
  - Activation choice
  - Cross validation evaluation
  - Reading from a csv

Possible improvements:
  - Reduce overhead from threading as the parameters must be passed as an object which must be instantiated
  - Reduce memory comsumption - each thread requires its own resources (activation values, partial derviatives)
  
