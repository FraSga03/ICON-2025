import sys
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

from plotter import plot_structure

def generate_random_example(m, num_samples=1):
    samples = m.simulate(num_samples)
    return samples

def discretize_column(data, column, bins, labels):
    data[column] = pd.cut(data[column], bins=bins, labels=labels)
    return data

def get_train_examples():
    data_train = pd.read_csv("./datasets/doncic_ref_train.csv")

    data_train = data_train.apply(pd.to_numeric, downcast='float')
    data_train = data_train.apply(pd.to_numeric, downcast='integer')

    data_train = discretize_column(data_train, "3P%", bins=[0.1, 0.2, 0.4, 0.5, 0.6], labels=['Low', 'Medium', 'High', 'Very High'])
    data_train = discretize_column(data_train, "GmSc", bins=[5, 10, 20, 30, 35], labels=['Low', 'Medium', 'High', 'Very High'])

    return pd.DataFrame(data_train)

def generate_learned_structure():
    np.random.seed(42)
    print("Creating structure...")
    data = get_train_examples()

    hill_climb = HillClimbSearch(data)
    best_model = hill_climb.estimate(scoring_method=K2Score(data), max_indegree=2, max_iter=200)

    bayesian_model = BayesianNetwork(best_model.edges())

    print("Structure created.")
    print("Fitting...")

    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)
    return bayesian_model

def generate_my_structure():
    bayesian_model = BayesianNetwork([
        ("3P%", "GmSc"),
        ("AST", "GmSc"),
        ("PTS", "GmSc"),
        ("TRB", "GmSc"),
        ("3P", "WIN"),
        ("3P%", "WIN"),
        ("AST", "WIN"),
        ("GmSc", "WIN"),
        ("PTS", "WIN"),
        ("TRB", "WIN")
    ])

    data_train = get_train_examples()

    bayesian_model.fit(data_train, estimator=MaximumLikelihoodEstimator)
    return bayesian_model


try:
    command = int(sys.argv[1])

    if command == 0:
        model = generate_learned_structure()
    else:
        model = generate_my_structure()

    model_name = "auto_model" if command == 0 else "my_model"

    print("Struttura imparata: \n", model)
    plot_structure(model, f"./documents/bayesian_networks/{model_name}.png")
    print("Esempi casuali: \n", generate_random_example(model, 20))
except:
    print("Parametri errati o mancanti")