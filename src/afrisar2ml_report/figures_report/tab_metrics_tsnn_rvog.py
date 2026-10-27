import torch

from afrisar2ml_report import config, metrics, open_dataset, paths
from afrisar2ml_report.benchmarks import benchmark_rvog, benchmark_tsnn


def get_metrics(prediction_list, target):
    rmse_list, bias_list, r2_list, pearson_list = [], [], [], []
    for prediction in prediction_list:
        valid_pixels = torch.isfinite(prediction) & torch.isfinite(target)
        metrics_dict = metrics.get_combined_metrics_dict(prediction[valid_pixels], target[valid_pixels])
        rmse_list.append(metrics_dict["rmse"])
        bias_list.append(metrics_dict["bias"])
        r2_list.append(metrics_dict["r2"])
        pearson_list.append(metrics_dict["pearson"])
    return torch.tensor(rmse_list), torch.tensor(bias_list), torch.tensor(r2_list), torch.tensor(pearson_list)


def _create_and_save_table(site, prediction_dict_p, prediction_dict_l, comment, save_to):
    # load lvis
    ds = open_dataset.open_dataset(site)
    target_pt = torch.tensor(open_dataset.get_non_water_rh98(ds))
    # get metrics
    metrics_str = f"% {comment} \n"
    metrics_str += "\\begin{tabular}{lccccc} \n"
    metrics_str += "\\hline \n"
    metrics_str += "\\textbf{Method} & \\textbf{Band} & \\textbf{RMSE (m)}"
    metrics_str += " & \\textbf{Bias (m)} & \\textbf{$R^2$} & \\textbf{Pearson} \\\\ \n"
    metrics_str += "\\hline \n"
    for band in ["P", "L"]:
        prediction_dict = prediction_dict_p if band == "P" else prediction_dict_l
        for method, prediction_list in prediction_dict.items():
            show_std = len(prediction_list) > 1
            metrics_str += f"{method} & {band}"
            rmse_list, bias_list, r2_list, pearson_list = get_metrics(prediction_list, target_pt)
            # RMSE
            metrics_str += f" & {torch.mean(rmse_list):.2f}"
            metrics_str += f" $\\pm$ {torch.std(rmse_list):.2f}" if show_std else ""
            # Bias
            metrics_str += f" & {torch.mean(bias_list):.2f}"
            metrics_str += f" $\\pm$ {torch.std(bias_list):.2f}" if show_std else ""
            # R2
            metrics_str += f" & {torch.mean(r2_list):.2f}"
            metrics_str += f" $\\pm$ {torch.std(r2_list):.2f}" if show_std else ""
            # Pearson
            metrics_str += f" & {torch.mean(pearson_list):.2f}"
            metrics_str += f" $\\pm$ {torch.std(pearson_list):.2f}" if show_std else ""
            metrics_str += " \\\\ \n"
        metrics_str += "\\hline \n"
    metrics_str += "\\end{tabular} \n"
    # save
    with open(save_to, "w") as file:
        file.write(metrics_str)


def tab_metrics_tsnn(site, seed, n_runs):
    print(f"tab_metrics_tsnn {site}, s{seed}n{n_runs}")
    prediction_dict_p = {
        "TSNN": [torch.tensor(benchmark_tsnn.load_tsnn_height(site, "P", seed + i)) for i in range(n_runs)],
    }
    prediction_dict_l = {
        "TSNN": [torch.tensor(benchmark_tsnn.load_tsnn_height(site, "L", seed + i)) for i in range(n_runs)],
    }
    comment = f"tab_metrics_tsnn, {site}, seed {seed}, {n_runs} runs"
    save_to = paths.get_report_figure_folder() / f"tab_metrics_tsnn_{site}_s{seed}n{n_runs}.tex"
    _create_and_save_table(site, prediction_dict_p, prediction_dict_l, comment, save_to)


def tab_metrics_rvog(site):
    print(f"tab_metrics_rvog {site}")
    pass_a_p, pass_b_p = benchmark_rvog.get_pass_names_by_site(site, "P")
    prediction_dict_p = {
        "RVoG": [torch.tensor(benchmark_rvog.load_rvog_height(site, "P", pass_a_p, pass_b_p))],
    }
    pass_a_l, pass_b_l = benchmark_rvog.get_pass_names_by_site(site, "L")
    prediction_dict_l = {
        "RVoG": [torch.tensor(benchmark_rvog.load_rvog_height(site, "L", pass_a_l, pass_b_l))],
    }
    comment = f"tab_metrics_rvog, {site}"    
    save_to = paths.get_report_figure_folder() / f"tab_metrics_rvog_{site}.tex"
    _create_and_save_table(site, prediction_dict_p, prediction_dict_l, comment, save_to)


if __name__ == "__main__":
    tab_metrics_tsnn(config.TEST_SITE, config.SEED, config.N_RUNS)
    tab_metrics_rvog(config.TEST_SITE)
