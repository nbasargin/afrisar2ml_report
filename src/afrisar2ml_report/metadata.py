"""
Helper functions to extract specific metadata from the dataset.
"""

from datetime import datetime


def get_pass_start_time(ds, ps_name, band):
    details = ds.attrs[f"pass_details_{band.lower()}"]
    ps_details = details[ps_name]
    return datetime.fromisoformat(ps_details["start_time"])


def get_pass_spatial_baselines(ds, ps_name, band):
    details = ds.attrs[f"pass_details_{band.lower()}"]
    ps_details = details[ps_name]
    h_bl, v_bl = ps_details["mean_horizontal_baseline_m"], ps_details["mean_vertical_baseline_m"]
    return h_bl, v_bl


def get_spatial_baselines_between(ds, ps_name_a, ps_name_b, band):
    h_bl_a, v_bl_a = get_pass_spatial_baselines(ds, ps_name_a, band)
    h_bl_b, v_bl_b = get_pass_spatial_baselines(ds, ps_name_b, band)
    h_bl = h_bl_b - h_bl_a
    v_bl = v_bl_b - v_bl_a
    return h_bl, v_bl
