import numpy as np
import torch
import torch.nn as nn
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scikitplot.metrics import plot_roc

nfeatures= 3

class band_gap_classifier(nn.Module):
    # initialazing the neural network with functions for the layers types using Linear transformation.
    def __init__(self, name: str = 'default classifier'):
        super(band_gap_classifier, self).__init__()
        self.name = name
        self.layer1 = nn.Linear(in_features=nfeatures, out_features=50)
        # self.layer1.weight = nn.Parameter(torch.zeros(self.layer1.weight.shape)) -> change the initial wights for each layer
        self.layer2 = nn.Linear(in_features=50, out_features=50)
        self.layer3 = nn.Linear(in_features=50, out_features=2)
        # this will set the desired activation function which will transfer layer to the next layer.
        self.activation = nn.Sigmoid()
        self.L3toProb = nn.Softmax(dim=1)
        # converting the output of the last layer to probability outputs, ! beware that this might not be nessesery using our cost func - BCE

    def forward(self, x: torch.Tensor):
        a1 = self.activation(self.layer1(x))
        a2 = self.activation(self.layer2(a1))
        a3 = self.activation(self.layer3(a2))
        y_pred = self.L3toProb(a3)
        return y_pred


def train(x_train: torch.TensorType, y_train: torch.TensorType, x_val: torch.TensorType, y_val: torch.TensorType):
    print("Enter Your Machine learning classifier name:")
    classiffier_name = input()
    cost_fn = nn.BCELoss()
    # we choose which cost function to use to train the model
    classifier = band_gap_classifier(name=classiffier_name )
    optimizer = torch.optim.SGD(params=classifier.parameters(), lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    # momentum=0.9, this parameter allows us to overcome local minima
    # parameters are heritage from Module class, with defult value set when initializing classifier
    # num_samples =y_train.shape[0]
    # y_train_ext = torch.ones([num_samples, 2])
    # vecArrange = torch.arange(num_samples)
    # y_train_ext[vecArrange, y_train[vecArrange]]=0
    num_iterations = 200
    train_cost_list = []
    val_cost_list = []
    for i in range(num_iterations):
        y_train_pred = classifier.forward(x_train)
        y_val_pred = classifier.forward(x_val)
        # using parenthesis after classifier directly calls the 'forward' function.
        train_cost = cost_fn(y_train_pred, y_train)
        val_cost = cost_fn(y_val_pred, y_val)
        val_cost_list.append(val_cost.item())
        train_cost_list.append(train_cost.item())
        optimizer.zero_grad()
        # we want to set the grads to zero because the way cost work is by accomulating the grads - more convineint for other neural networks
        train_cost.backward()
        # print('iteration: ', i, 'train cost: ', train_cost.item())
        optimizer.step()
        # this code is counter intuitive- this functions run in the background and change the values of the wight parameters and set a gradient in each step.
        scheduler.step()
    return classifier, train_cost_list, val_cost_list


def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["nonmetal", "metal"])
    disp.plot()
    plt.show()


def plot_ROC(y_test, y_pred_prob):
    fig, ax = plt.subplots()
    plot_roc(y_test, y_pred_prob, ax=ax, plot_macro=False)
    plt.show()


def get_performance(y_pred: torch.Tensor, y_pred_prob, y_test: torch.Tensor, classifier_name):
    nsamples = y_pred.size(dim=0)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(nsamples):
        if y_pred[i] == y_test[i]:
            if y_pred[i] == 1:
                TP += 1
            else:
                TN += 1
        elif y_pred[i] == 1:
            FP += 1
        else:
            FN += 1
    plot_confusion(y_test, y_pred)
    plot_ROC(y_test, y_pred_prob)
    accuracy = (TP + TN) / nsamples
    FPR = FP / (TP + FN)
    FNR = FN / (TP + FN)
    TNR = TN / (TN + FP)
    NPV = TN / (TN + FN)
    FDR = FP / (FP + TP)
    TPR_recall = TP / (TP + FN)
    PPV_precision = TP / (TP + FP)
    F1 = 2 * PPV_precision * TPR_recall / (PPV_precision + TPR_recall)
    with open("classifiers_performances.csv", 'a', encoding='UTF8', newline='') as file:
        f = csv.writer(file)
        f.writerow(['Classifier Name', 'Accuracy', 'TPR_Recall', 'PPV_Precision ', 'F1', 'FPR', 'FNR', 'TNR', 'NPV',
                    'FDR', 'TP', 'TN', 'FP', 'FN'])
        f.writerow([classifier_name, accuracy.__round__(3), TPR_recall.__round__(3), PPV_precision.__round__(3),
                    F1.__round__(3), FPR.__round__(3), FNR.__round__(3), TNR.__round__(3), NPV.__round__(3),
                    FDR.__round__(3), TP, TN, FP, FN])
    # total_hits = torch.sum(torch.abs(y_pred-y_test_new))
    # total_hits.item()
    # this line of code is an example to how a command should be executed using pytorch, operating on tensors.


def test(classifier: band_gap_classifier, x_test: torch.TensorType, y_test: torch.TensorType):
    y_pred_prob = classifier.forward(x_test)
    y_pred = y_pred_prob.argmax(dim=1)
    y_test_new = y_test.argmax(dim=1)
    test_eq_pred = torch.sum(torch.abs(y_pred - y_test_new)) / y_test.size(dim=0)
    # this line of code is an example to how a command should be executed using pytorch, operating on tensors.
    get_performance(y_pred, y_pred_prob.detach().numpy(), y_test_new, classifier.name)
