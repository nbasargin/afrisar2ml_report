import torch
from torch import Tensor
from torchmetrics.functional import mean_squared_error, pearson_corrcoef, r2_score


def get_rmse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mean_squared_error(prediction, target))


def get_bias(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(prediction - target)


def get_r2(prediction: Tensor, target: Tensor) -> Tensor:
    return r2_score(prediction, target)


def get_combined_metrics_dict(prediction: Tensor, target: Tensor):
    rmse = get_rmse(prediction, target)
    bias = get_bias(prediction, target)
    pearson = pearson_corrcoef(prediction, target)
    r2 = get_r2(prediction, target)
    return {"rmse": rmse, "bias": bias, "pearson": pearson, "r2": r2}
