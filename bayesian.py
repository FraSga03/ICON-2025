import sys
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, K2Score, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.models import BayesianNetwork

from plotter import plot_structure

def generate_random_example(model, num_samples=1):
    samples = model.simulate(num_samples)
    return samples

def discretize_column(data, column, bins, labels):
    data[column] = pd.cut(data[column], bins=bins, labels=labels)
    return data

def get_train_examples(n):
    data_train = pd.read_csv("./datasets/doncic_ref_train.csv").sample(n)

    data_train = data_train.apply(pd.to_numeric, downcast='float')
    data_train = data_train.apply(pd.to_numeric, downcast='integer')

    data_train = discretize_column(data_train, "3P%", bins=[0.1, 0.2, 0.4, 0.5, 0.6], labels=['Low', 'Medium', 'High', 'Very High'])
    data_train = discretize_column(data_train, "GmSc", bins=[5, 10, 20, 30, 35], labels=['Low', 'Medium', 'High', 'Very High'])

    return pd.DataFrame(data_train)


def generate_learned_structure():
    np.random.seed(42)
    print("Creating structure...")
    data = get_train_examples(25)

    hill_climb = HillClimbSearch(data)
    best_model = hill_climb.estimate(scoring_method=K2Score(data), max_indegree=3, max_iter=100)

    bayesian_model = BayesianNetwork(best_model.edges())

    print("Structure created.")
    print("Fitting...")

    bayesian_model.fit(data, estimator=MaximumLikelihoodEstimator)

    print("Learned Structure: ", bayesian_model)
    plot_structure(bayesian_model)

    print("Random Examples: ", generate_random_example(bayesian_model, 5))

def test_my_structure():
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

    data_train = get_train_examples(100)

    bayesian_model.fit(data_train, estimator=MaximumLikelihoodEstimator)

    print("Learned Structure: ", bayesian_model)
    plot_structure(bayesian_model)

    print("Random Examples: \n", generate_random_example(bayesian_model, 5))

command = int(sys.argv[1])

if command == 0:
    generate_learned_structure()
else:
    test_my_structure()

