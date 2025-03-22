import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from plotter import plot_conf_matrix, plot_roc_curve, plot_pr_rate_curve, plot_f1

class SupervisedModel(object):
    model = ""
    params = ""
    must_normalize = False

    def __init__(self, model, params, must_normalize):
        self.model = model
        self.params = params
        self.must_normalize = must_normalize


def generate_report(model, x_test, y_test, name, training_type):
    print("Stampa report...")

    y_prediction = model.predict(x_test)

    plot_conf_matrix(y_test, y_prediction, name, f"documents/{type(model).__name__}/{training_type}_MATRIX.png")
    plot_roc_curve(y_test, y_prediction, name, f"documents/{type(model).__name__}/{training_type}_ROC.png")
    plot_pr_rate_curve(y_test, y_prediction, name, f"documents/{type(model).__name__}/{training_type}_PR.png")
    plot_f1(y_test, y_prediction, name, f"documents/{type(model).__name__}/{training_type}_F1.png")

    print(f"Accuratezza {accuracy_score(y_test, y_prediction)}\n")


def dt_with_grid_k_fold_shuffle(model, params, x_train, y_train, x_test, y_test, must_normalize):
    print(f"Training {type(model).__name__} con KFold e shuffle")

    if must_normalize:
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)

    kf = KFold(n_splits=10, shuffle=True, random_state=40)

    print("Scelta parametri attraverso greedy search:")

    grid_search = GridSearchCV(model, param_grid=params, cv=kf, scoring="accuracy", n_jobs=1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    print(grid_search.best_params_)
    print("Training finito")

    return {
        generate_report(
            best_model,
            x_test,
            y_test,
            f"{type(model).__name__} con KFold e shuffle",
            f"KF_SHUFFLE"
        ),
        best_model
    }

def dt_with_grid_k_fold(model, params, x_train, y_train, x_test, y_test, must_normalize):
    print(f"Training {type(model).__name__} con KFold senza shuffle")

    if must_normalize:
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)

    print("Scelta parametri attraverso greedy search:")

    grid_search = GridSearchCV(model, param_grid=params, cv=10, scoring="accuracy", n_jobs=1)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_

    print(grid_search.best_params_)
    print("Training finito")

    return {
        generate_report(
            best_model,
            x_test,
            y_test,
            f"{type(model).__name__}  con KFold e senza shuffle",
            f"KF_WO_SHUFFLE"
        ),
        best_model
    }


def dt_without_cv(model, x_train, y_train, x_test, y_test, must_normalize):
    print(f"Training {type(model).__name__} senza KF")

    if must_normalize:
        scaler = StandardScaler()

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)

    model.fit(x_train, y_train)

    print("Training finito")

    return {
        generate_report(
            model,
            x_test,
            y_test,
            f"{type(model).__name__} senza CV",
            "WO_KF"
        ),
        model
    }


data_train = pd.read_csv("./datasets/doncic_ref_train.csv")
data_test = pd.read_csv("./datasets/doncic_ref_test.csv")

# remove target feature
X_train = data_train.drop(columns=["WIN"])
X_test = data_test.drop(columns=["WIN"])

# pick target feature
Y_train = data_train["WIN"]
Y_test = data_test["WIN"]

predictors = [
    SupervisedModel(
        DecisionTreeClassifier(random_state=40),
        {
            'max_depth': [3, 5],  # Profondità dell'albero
            'min_samples_split': [2, 5],  # Numero minimo di campioni per dividere un nodo
            'min_samples_leaf': [1, 2],  # Numero minimo di campioni in una foglia
            'criterion': ['gini', 'entropy']  # Metodo di split
        },
        False
    ),
    SupervisedModel(
        RandomForestClassifier(random_state=40),
        {
            'n_estimators': [10, 15, 20], # Numero di alberi
            'max_depth': [10, 20], # Profondità massima degli alberi
            'min_samples_split': [2, 5, 10],  # Minimo campioni per dividere
            'min_samples_leaf': [1, 2, 4]  # Minimo campioni per foglia
        },
        False
    ),
    SupervisedModel(
        SVC(kernel='linear'),
        {
            'C': [0.1, 1, 10],  # Parametro di penalizzazione
            'kernel': ['linear', 'rbf'],  # Tipo di kernel
            'gamma': ['scale', 'auto', 0.1, 1]  # Parametro per il kernel RBF
        },
        True
    ),
    SupervisedModel(
        LogisticRegression(random_state=40),
        {
            'penalty': ['l1','l2'],
            'C': [0.001, 0.01, 0.1],
            'solver': ['liblinear'],
            'max_iter': [100000, 150000],
        },
        True
    ),
    SupervisedModel(
        KNeighborsClassifier(),
        {
            'n_neighbors': [2, 3, 5],
            'weights': ['uniform', 'distance'],
            'algorithm': ["brute"],
            'p': [1, 2, 4],
        },
        True
    )
]

try:
    index = int(sys.argv[1])
    predictor = predictors[index]

    acc1, model1 = dt_without_cv(predictor.model, X_train, Y_train, X_test, Y_test, predictor.must_normalize)
    acc2, model2 = dt_with_grid_k_fold(predictor.model, predictor.params, X_train, Y_train, X_test, Y_test, predictor.must_normalize)
    acc3, model3 = dt_with_grid_k_fold_shuffle(predictor.model, predictor.params, X_train, Y_train, X_test, Y_test, predictor.must_normalize)
except:
    print("Indica un modello utilizzando un parametro che va da 0 a 4")
