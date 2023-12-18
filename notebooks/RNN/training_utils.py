from uuid import uuid4
import json
import os
import torch


def save_model(model, model_save_dir, step, model_metrics):

    model_save_dir = os.path.join(
        model_save_dir, str(uuid4()), f"checkpoint-{step}")

    model_name = "pytorch_model.pt"
    train_state_name = "training_state.json"
    os.makedirs(model_save_dir, exist_ok=True)

    model_path = os.path.join(model_save_dir, model_name)
    train_state_path = os.path.join(model_save_dir, train_state_name)

    torch.save(model.state_dict(), model_path)

    if model_metrics is not None:
        metrics_dict = {}
        for key, value in model_metrics.items():
            metrics_dict[key] = float(value)

        with open(train_state_path, mode='w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)

    return model_path
