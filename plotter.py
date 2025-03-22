import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report


def plot_conf_matrix(y_test, y_prediction, name, filename):
    conf_matrix = confusion_matrix(y_test, y_prediction)
    plot.figure(figsize=(6,5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predetti Falsi', 'Predetti Veri'],
        yticklabels=['Falsi', 'Veri']
    )
    plot.ylabel("Valori Reali")
    plot.xlabel("Valori Predetti")
    plot.title(name)

    plot.savefig(filename)

    plot.show()


def plot_roc_curve(y_test, y_prediction, name, filename):
    fpr, tpr, thresholds = roc_curve(y_test, y_prediction)
    roc_auc = auc(fpr, tpr)

    plot.figure(figsize=(6,5))
    plot.plot(fpr, tpr, color='green', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plot.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.legend(loc='best')
    plot.title(name)

    plot.savefig(filename)

    plot.show()


def plot_pr_rate_curve(y_test, y_prediction, name, filename):
    p, r, _ = precision_recall_curve(y_test, y_prediction)

    plot.figure(figsize=(6,5))
    plot.plot(r, p, color="green", lw=2)
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.title('Precision-Recall Curve')
    plot.title(name)

    plot.savefig(filename)

    plot.show()

def plot_f1(y_test, y_prediction, name, filename):
    report = classification_report(y_test, y_prediction, output_dict=True)
    f1_scores = [report[str(i)]['f1-score'] for i in np.unique(y_test)]
    classes = [str(i) for i in np.unique(y_test)]

    plot.figure(figsize=(6, 4))
    plot.bar(classes, f1_scores, color=['blue', 'green'])
    plot.xlabel("Class")
    plot.ylabel("F1 Score")
    plot.title("F1 Score per Class")
    plot.ylim(0, 1)
    plot.title(name)

    plot.savefig(filename)

    plot.show()


def plot_structure(model):
    graph = nx.DiGraph()
    graph.add_edges_from(model.edges())

    plot.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42, k=10)
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="green", edge_color="black", font_size=10)
    plot.title("Learned Bayesian Network Structure")
    plot.show()