def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    iris = load_iris()
    scoreModel = None
    # YOUR CODE HERE
    X = iris.data
    y = iris.target
    from sklearn.tree import DecisionTreeClassifier
    dtmodel = DecisionTreeClassifier(random_state=0)
    dtmodel.fit(X,y)
    scoreModel = dtmodel.score(X,y)

    return scoreModel


if __name__ == '__main__':
    print(homework())
