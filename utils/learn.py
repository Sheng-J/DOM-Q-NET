import torch
import ipdb


def epoch_train(
	loss_f, optimizer, train_loader
	):
    """
    loss_f (inputs) from dataset
    """
    num_batches = 0
    cumu_loss = 0.0
    for inputs in train_loader:
        loss = loss_f(*inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cumu_loss += loss.item()
        num_batches += 1
    epoch_loss = cumu_loss / num_batches
    return epoch_loss


