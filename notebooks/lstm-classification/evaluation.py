import torch
from torch import nn
import numpy as np
from sklearn import metrics
from tqdm import tqdm


def get_binary_metric(labels_list, pred_list, prob_list, train_or_eval="eval"):

    accuracy = metrics.accuracy_score(labels_list, pred_list)
    recall = metrics.recall_score(labels_list, pred_list)
    precision = metrics.precision_score(labels_list, pred_list)
    f1 = metrics.f1_score(labels_list, pred_list)
    fpr, tpr, _ = metrics.roc_curve(
        labels_list, prob_list, pos_label=1)

    auc = metrics.auc(fpr, tpr)

    result = {f"{train_or_eval}_accuracy": accuracy,
              f"{train_or_eval}_recall": recall,
              f"{train_or_eval}_precision": precision,
              f"{train_or_eval}_recall": recall,
              f"{train_or_eval}_f1": f1,
              f'{train_or_eval}_auc': auc}

    return result


def evaluate(model, test_loader, criterion, train_or_eval="eval"):

    model.eval()

    model.eval()
    loss_list = []
    labels_list = []
    pred_list = []
    prob_list = []

    pbar = tqdm(total=len(test_loader),
                desc="Evaluating:", position=0, leave=True)

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_list.append(loss.item())

            yhat = torch.argmax(outputs, 1).cpu().numpy()
            pred_list.extend(yhat)

            y = targets.cpu().numpy()
            labels_list.extend(y)

            prob = torch.sigmoid(outputs[:, 1])
            prob = prob.cpu().numpy()
            prob_list.extend(prob)

            pbar.update(1)
        pbar.close()

    loss = np.mean(loss_list)

    metrics_results = get_binary_metric(
        labels_list, pred_list, prob_list, train_or_eval)

    metrics_results[f"{train_or_eval}_loss"] = loss

    return metrics_results, labels_list, pred_list, prob_list
