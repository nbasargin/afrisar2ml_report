import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from afrisar2ml_report import hardware, metrics
from afrisar2ml_report.tsnn.tsnn_dataset import TSNNDataset
from afrisar2ml_report.tsnn.tsnn_model import TSNNnet9LRegression, tsnn_mpmb_matrix_to_feature_vector


def run_tsnn_on_ds(model: TSNNnet9LRegression, ds: TSNNDataset, device, batch_size=1024):
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    prediction_batches = []
    height_batches = []
    iters = len(loader)
    for i, (mpmb, height) in enumerate(loader):
        feats = tsnn_mpmb_matrix_to_feature_vector(mpmb)
        feats = feats.to(device)
        height = height.to(device)
        predicted = model.forward(feats)
        prediction_batches.append(predicted.cpu().detach())
        height_batches.append(height.cpu().detach())
        if (i + 1) % 200 == 0:
            print(f"Iter {i + 1:>5} / {iters}")
    predictions = torch.concat(prediction_batches)
    targets = torch.concat(height_batches)
    return predictions, targets


def train_new_tsnn_model(ds_train: TSNNDataset, ds_val: TSNNDataset | None, device, n_epochs, eval_epochs):
    _n_samples, np1, _np2 = ds_train.mpmb.shape
    n_pols = 4
    n_tracks = np1 // n_pols
    # Constants from the TSNN paper
    TSNN_N_NODES = 400
    TSNN_LR = 1e-4
    TSNN_BETA_1 = 0.9
    TSNN_BETA_2 = 0.999
    TSNN_BATCH_SIZE = 32
    # prepare training
    model = TSNNnet9LRegression(n_tracks=n_tracks, n_pols=n_pols, n_nodes=TSNN_N_NODES)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=TSNN_LR, betas=(TSNN_BETA_1, TSNN_BETA_2))
    criterion = nn.MSELoss()
    loss_history = []
    val_metrics_by_epoch = {}
    loader = DataLoader(
        ds_train,
        batch_size=TSNN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    iters = len(loader)

    # start training
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} / {n_epochs}")
        hardware.print_ram_usage()
        hardware.print_cuda_memory_usage()
        model.train()
        for i, (mpmb, height) in enumerate(loader):
            optimizer.zero_grad()
            feats = tsnn_mpmb_matrix_to_feature_vector(mpmb)
            feats = feats.to(device)
            height = height.to(device)
            prediction = model.forward(feats)
            loss = criterion(prediction, height)
            loss.backward()
            optimizer.step()
            # stats
            loss_item = loss.item()
            loss_history.append(loss_item)
            if (i + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1} / {n_epochs}  |  Iter {i + 1:>5} / {iters}  |  Loss {loss_item:<10.4f}")
        if (epoch + 1) in eval_epochs and ds_val is not None:
            predicted, target = run_tsnn_on_ds(model, ds_val, device)
            metrics_dict = metrics.get_combined_metrics_dict(predicted, target)
            val_metrics_by_epoch[epoch + 1] = metrics_dict
    return model, loss_history, val_metrics_by_epoch
