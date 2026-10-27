import numpy as np
import torch

from afrisar2ml_report import config, open_dataset


def split_data_into_train_val_test(height_dict, band):
    """
    Subdivide height values from all sites into train / validation / test sets.
    Also load matching LVIS RH98 metrics.

    height_dict is a dict that contains height predictions for each site.
    """
    # train and validation areas across several sites
    train_pred_list = []
    train_trg_list = []
    val_pred_list = []
    val_trg_list = []
    for site in config.TRAIN_VAL_SITES:
        predictions = height_dict[site]
        ds = open_dataset.open_dataset(site)
        rh98 = open_dataset.get_non_water_rh98(ds)
        mask_fsar = ds[f"mask_{band.lower()}"].values.astype(np.bool)
        mask_subregion = ds["mask_subregion"].values.astype(np.bool)
        mask_lvis = np.isfinite(rh98)
        mask_predictions = np.isfinite(predictions)
        train_mask = mask_fsar & mask_lvis & mask_predictions & (~mask_subregion)
        val_mask = mask_fsar & mask_lvis & mask_predictions & mask_subregion
        train_pred_list.append(predictions[train_mask])
        val_pred_list.append(predictions[val_mask])
        train_trg_list.append(rh98[train_mask])
        val_trg_list.append(rh98[val_mask])
    train_pred = torch.tensor(np.concat(train_pred_list))
    train_trg = torch.tensor(np.concat(train_trg_list))
    val_pred = torch.tensor(np.concat(val_pred_list))
    val_trg = torch.tensor(np.concat(val_trg_list))

    # test area, another independent full site
    test_pred_all = height_dict[config.TEST_SITE]
    ds_test = open_dataset.open_dataset(config.TEST_SITE)
    rh98_test = open_dataset.get_non_water_rh98(ds_test)
    mask_fsar_test = ds_test[f"mask_{band.lower()}"].values.astype(np.bool)
    mask_lvis_test = np.isfinite(rh98_test)
    mask_predictions_test = np.isfinite(test_pred_all)
    test_mask = mask_fsar_test & mask_lvis_test & mask_predictions_test
    test_pred = torch.tensor(test_pred_all[test_mask])
    test_trg = torch.tensor(rh98_test[test_mask])

    return train_pred, train_trg, val_pred, val_trg, test_pred, test_trg
