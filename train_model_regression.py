#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Training functions for PIMS-Net regression task (Sea Ice Concentration prediction)."""

# -- File info -- #
__author__ = 'Xinyi Liu, Wanxia Deng, Lei Liu, Jinfeng Ding, Xiao Cheng'
__contributor__ = ''
__copyright__ = ['National University of Defense Technology School of Meteorology and Oceanology']
__contact__ = ['liuxinyi20@nudt.edu.cn', 'dengwanxia14@nudt.edu.cn']
__version__ = '1.0.0'
__date__ = '2026-04-30'

# -- Built-in modules -- #
import copy
import json

# -- Third-party modules -- #
import torch

# -- Proprietary modules -- #
from utils.functions import r2_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, optimizer, traindataloader, valdataloader, train_options, scheduler):
    """
        Train a regression model (PIMS-Net) for Sea Ice Concentration prediction.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to train.
        optimizer : torch.optim.Optimizer
            Optimizer for updating weights.
        traindataloader : torch.utils.data.DataLoader
            DataLoader for training data.
        valdataloader : torch.utils.data.DataLoader
            DataLoader for validation data.
        train_options : dict
            Dictionary containing training hyperparameters (epochs, charts, etc.).
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.

        Returns
        -------
        torch.nn.Module
            Model with the best weights (lowest validation MSE loss).
        """
    best_loss = 1e10
    best_model_wts = copy.deepcopy(model.state_dict())

    # Use Mean Squared Error (MSE) loss for regression
    loss_functions = {chart: torch.nn.MSELoss() for chart in train_options['charts']}

    for epoch in range(train_options['epochs']):
        print('Epoch {}/{}'.format(epoch + 1, train_options['epochs']))
        train_loss = 0.0
        train_num = 0

        # Accumulators for training metrics
        train_metrics_accumulator = {metric: 0.0 for metric in ["R2 Score", "MSE Loss"]}
        val_metrics_accumulator = {metric: 0.0 for metric in ["R2 Score", "MSE Loss"]}

        # ---------- Training Phase ----------
        model.train()
        for step, (b_x, b_y, _, masks) in enumerate(traindataloader):
            optimizer.zero_grad()
            loss_batch = 0

            # b_x = torch.cat((b_x[:, :2, ...], b_x[:, 4:, ...]), dim=1).float().to(device)
            # b_x = b_x.float().to(device)
            # output = model(b_x)

            # Split multi-modal inputs (TB, SAR, MET)
            bt = b_x[:, 4:18, :, :].float().to(device)
            sar = b_x[:, :2, :, :].float().to(device)
            met = b_x[:, 18:, :, :].float().to(device)
            output = model(bt, sar, met)

            # 遍历所有图表
            for chart in train_options['charts']:
                # Select valid pixels (non-masked)
                valid_output = output.squeeze()[~masks[chart]]
                valid_label = b_y[chart][~masks[chart]].to(device) / 10
                loss_chart = loss_functions[chart](input=valid_output, target=valid_label.float()).float()
                loss_batch += loss_chart
                train_metrics_accumulator["MSE Loss"] += loss_chart.item()

                # Compute R2 score for this batch
                batch_metrics = r2_metric((valid_label).cpu().numpy(), valid_output.detach().cpu().numpy())

                for metric, value in batch_metrics.items():
                    train_metrics_accumulator[metric] += value
            # Backpropagation
            loss_batch.backward()
            optimizer.step()
            train_loss += loss_batch.item() * len(b_x)
            train_num += len(b_x)

        # Log training loss and metrics
        train_loss_epoch = train_loss / train_num
        print(f"Epoch {epoch + 1} Train Loss: {train_loss_epoch:.4f}")
        train_metrics_average = {metric: value / len(traindataloader) for metric, value in train_metrics_accumulator.items()}
        print(f"Epoch {epoch + 1} Train Metrics:")
        for metric, avg_value in train_metrics_average.items():
            print(f"  {metric}: {avg_value:.3f}")

        # update_json_file('results/PIMS_Net/random_2_train.json', {f"epoch {epoch + 1}": train_metrics_average})

        # Free memory
        del b_x, b_y, _, masks, output, valid_output, valid_label, batch_metrics, loss_chart, loss_batch, \
            train_metrics_accumulator, train_loss, train_num, train_loss_epoch, train_metrics_average
        torch.cuda.empty_cache()

        # ---------- Validation Phase ----------
        model.eval()
        num_batches = 0

        with torch.no_grad():
            for step, (b_x, b_y, b_name, masks) in enumerate(valdataloader):

                # b_x = torch.cat((b_x[:, :2, ...], b_x[:, 4:, ...]), dim=1).float().to(device)
                # b_x = b_x.float().to(device)
                # output = model(b_x)

                bt = b_x[:, 4:18, :, :].float().to(device)
                sar = b_x[:, :2, :, :].float().to(device)
                met = b_x[:, 18:, :, :].float().to(device)
                output = model(bt, sar, met)

                for chart in train_options['charts']:
                    valid_output = output.squeeze()[~masks[chart]]
                    valid_label = b_y[chart][~masks[chart]].to(device) / 10

                    batch_metrics = r2_metric((valid_label).cpu().numpy(), valid_output.detach().cpu().numpy())

                    MSE_loss = loss_functions[chart](input=valid_output, target=valid_label.float()).float()
                    val_metrics_accumulator["MSE Loss"] += MSE_loss.item()

                    for metric, value in batch_metrics.items():
                        val_metrics_accumulator[metric] += value
                    num_batches += 1

            # Average validation metrics
            val_metrics_average = {metric: value / num_batches for metric, value in val_metrics_accumulator.items()}
            current_loss = val_metrics_average["MSE Loss"]

            print(f"Epoch {epoch + 1} Validation Metrics:")
            for metric, avg_value in val_metrics_average.items():
                print(f"  {metric}: {avg_value:.3f}")

            # update_json_file('results/PIMS_Net/random_2_val.json', {f"epoch {epoch + 1}": val_metrics_average})

            # Save best model based on validation MSE loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model found with MSE Loss: {best_loss:.4f}, saving model...")

            del b_x, b_y, b_name, masks, output, valid_output, valid_label, batch_metrics, \
                MSE_loss, val_metrics_accumulator, val_metrics_average, current_loss, num_batches
        print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        scheduler.step()
        torch.cuda.empty_cache()
    # Load the best model weights and return
    model.load_state_dict(best_model_wts)
    return model

def update_json_file(file_path, new_data):
    """
        Update a JSON file with new data (used for logging metrics).

        Parameters
        ----------
        file_path : str
            Path to the JSON file.
        new_data : dict
            Dictionary containing new data to merge into the JSON file.
        """
    try:
        with open(file_path, 'r') as f:
            epoch_metrics = json.load(f)
    except FileNotFoundError:
        epoch_metrics = {}
    epoch_metrics.update(new_data)

    with open(file_path, 'w') as f:
        json.dump(epoch_metrics, f, indent=4)