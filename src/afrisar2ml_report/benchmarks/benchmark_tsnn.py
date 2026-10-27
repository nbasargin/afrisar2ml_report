import datetime
import socket

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from afrisar2ml_report import config, hardware, paths, sites
from afrisar2ml_report.plot_utils import ml_plots
from afrisar2ml_report.tsnn import tsnn_dataset, tsnn_model, tsnn_train

N_TRACKS = 13  # fixed number of tracks, model assumes a fixed feature vector size


def _get_site_str(site_list: list[str]):
    return "+".join([s[0:3].lower() for s in site_list])


def _tsnn_model_path(band, seed):
    site_str = _get_site_str(config.TRAIN_VAL_SITES)
    return paths.get_cache_folder() / f"tsnn_model_{site_str}_{band}_v{config.REPORT_VERSION}s{seed}.pth"


def _tsnn_height_path(site, band, seed):
    return paths.get_cache_folder() / f"tsnn_{site}_{band}_height_v{config.REPORT_VERSION}s{seed}.npy"


def _hardware_setup(seed, device: torch.device):
    setup = "TSNN training setup\n"
    setup += f"Seed: {seed}\n"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device.index)
        setup += f"Device: {device} {gpu_name}\n"
    else:
        setup += f"Device: {device}\n"
    setup += f"Hostname: {socket.gethostname()}\n"
    setup += f"Date: {datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()}\n"
    setup += f"Numpy version: {np.__version__}\n"
    setup += f"Torch version: {torch.__version__}\n"
    return setup


def get_tsnn_train_val_datasets(band):
    # load and merge individual train/val sites
    ds_train_list = []
    ds_val_list = []
    for site in config.TRAIN_VAL_SITES:
        mpmb, height, mask_subregion = tsnn_dataset.load_tsnn_site_data(site, band, N_TRACKS)
        ds_train_i = tsnn_dataset.convert_data_to_dataset(mpmb, height, mask_subregion, site, band, "train")
        ds_val_i = tsnn_dataset.convert_data_to_dataset(mpmb, height, mask_subregion, site, band, "validation")
        ds_train_list.append(ds_train_i)
        ds_val_list.append(ds_val_i)
    site_str = _get_site_str(config.TRAIN_VAL_SITES)
    ds_train = tsnn_dataset.merge_tsnn_datasets(ds_train_list, f"ds_{site_str}_{band}_train")
    ds_val = tsnn_dataset.merge_tsnn_datasets(ds_val_list, f"ds_{site_str}_{band}_val")
    return ds_train, ds_val


def train_tsnn_for_benchmark(ds_train, ds_val, band, seed, device):
    torch.manual_seed(seed)
    model_path = _tsnn_model_path(band, seed)
    if model_path.exists():
        print(f"TSNN model for {band}, {seed} is already cached")
        return
    print(f"Training TSNN model for {band}, seed {seed}, on {device}")
    hardware.print_ram_usage()
    model_id = model_path.stem
    model_folder = model_path.parent
    n_epochs = 20  # less epochs than in the original papers, since more epochs leads to overfitting
    eval_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    model, loss_hist, val_metrics_by_epoch = tsnn_train.train_new_tsnn_model(
        ds_train, ds_val, device, n_epochs, eval_epochs
    )
    model.model_id = model_id
    # save model and stats
    tsnn_model.save_tsnn(model, model_path)
    training_stats = {
        "model_id": model_id,
        "train_ds_id": ds_train.ds_id,
        "val_ds_id": ds_val.ds_id,
        "loss_hist": loss_hist,
        "val_metrics_by_epoch": val_metrics_by_epoch,
    }
    torch.save(training_stats, model_folder / f"{model_id}__training_stats.pth")
    # plot stats
    ml_plots.plot_train_loss(loss_hist, save_to=model_folder / f"{model_id}__loss_hist.png", loss_max=100)
    ml_plots.plot_eval_metrics(
        val_metrics_by_epoch,
        f"Model {model_id}, validation dataset {ds_val.ds_id}",
        model_folder / f"{model_id}__val_metrics_by_epoch.png",
    )
    # save system setup
    with open(model_folder / f"{model_id}__system_setup.txt", "w") as file:
        file.write(_hardware_setup(seed, device))


def benchmark_tsnn_on_dataset(model: tsnn_model.TSNNnet9LRegression, ds: tsnn_dataset.TSNNDataset, device):
    model.eval()
    model.to(device)
    batch_size = 1024
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=hardware.is_cuda_available(),
        drop_last=False,
    )
    prediction_batches = []
    height_batches = []
    iters = len(loader)
    for i, (mpmb, height) in enumerate(loader):
        feats = tsnn_model.tsnn_mpmb_matrix_to_feature_vector(mpmb)
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


def benchmark_tsnn_on_site(model: tsnn_model.TSNNnet9LRegression, mpmb, device):
    model.eval()
    model.to(device)
    rows, cols, *_ = mpmb.shape
    predictions = torch.full((rows, cols), dtype=torch.float32, fill_value=torch.nan)
    valid_mask = torch.isfinite(mpmb[:, :, 0, 0])
    for row in range(rows):
        mpmb_row, valid_mask_row = mpmb[row], valid_mask[row]
        if torch.sum(valid_mask_row) == 0:
            continue
        mpmb_val = mpmb_row[valid_mask_row].to(device)
        feats = tsnn_model.tsnn_mpmb_matrix_to_feature_vector(mpmb_val)
        height_val = model.forward(feats).cpu().detach()
        predictions[row][valid_mask_row] = height_val
    return predictions


def cache_tsnn_height(site, band, seed, device):
    height_path = _tsnn_height_path(site, band, seed)
    if height_path.exists():
        print(f"TSNN canopy height for {site}, {band}, {seed} is already cached")
        return
    print(f"Caching TSNN canopy height for {site}, {band}, {seed}")
    # load data
    mpmb, _target, _mask_subregion = tsnn_dataset.load_tsnn_site_data(site, band, N_TRACKS)
    # load model
    model = tsnn_model.load_tsnn(_tsnn_model_path(band, seed), device)
    # run model on the whole site, save results
    height = benchmark_tsnn_on_site(model, mpmb, device)
    np.save(height_path, height.cpu().detach().numpy())


def load_tsnn_height(site, band, seed):
    return np.load(_tsnn_height_path(site, band, seed))


def main_benchmark_tsnn():
    matplotlib.use("Agg")
    print("Running TSNN benchmarks")
    device = hardware.get_device()
    # run model training
    for band in ["L", "P"]:
        ds_train, ds_val = get_tsnn_train_val_datasets(band)
        for run_i in range(config.N_RUNS):
            seed = config.SEED + run_i
            train_tsnn_for_benchmark(ds_train, ds_val, band, seed, device)

    # cache height predictions
    for site in [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]:
        for band in ["L", "P"]:
            for run_i in range(config.N_RUNS):
                seed = config.SEED + run_i
                cache_tsnn_height(site, band, seed, device)


if __name__ == "__main__":
    main_benchmark_tsnn()
