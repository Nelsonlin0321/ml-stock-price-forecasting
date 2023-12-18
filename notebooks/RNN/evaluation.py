import torch
from torch import nn
import numpy as np
from sklearn import metrics
from tqdm import tqdm


def evaluate(model, test_loader, scaler, criterion=nn.MSELoss()):

    model.eval()

    prediction_list = []
    ground_truth_list = []
    eval_loss_list = []
    price_loss_list = []

    pbar = tqdm(total=len(test_loader),
                desc="Evaluating:", position=0, leave=True)

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            price_loss = criterion(outputs[:, -1], targets[:, -1])

            yhat = outputs[:, -1].cpu().numpy()
            prediction_list.append(yhat)

            y = targets[:, -1].cpu().numpy()
            ground_truth_list.append(y)

            eval_loss_list.append(loss.item())
            price_loss_list.append(price_loss.item())

            pbar.update(1)
        pbar.close()

    predictions = np.concatenate(prediction_list)
    ground_truths = np.concatenate(ground_truth_list)

    predictions = np.exp(scaler.inverse_transform(
        predictions.reshape(-1, 1)))[:, 0]
    ground_truths = np.exp(scaler.inverse_transform(
        ground_truths.reshape(-1, 1)))[:, 0]

    mae = metrics.mean_absolute_error(ground_truths, predictions)

    eval_metrics = {}
    eval_metrics['eval_loss'] = sum(eval_loss_list)/len(eval_loss_list)
    eval_metrics['eval_price_loss'] = sum(price_loss_list)/len(price_loss_list)
    eval_metrics['eval_MEA'] = mae

    return eval_metrics
