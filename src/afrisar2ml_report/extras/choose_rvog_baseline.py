"""
List the primary-secondary pass pairs matching the specified baseline.
"""

import numpy as np

from afrisar2ml_report import metadata, open_dataset, sites


def pass_lists_by_site(site):
    return {
        sites.LOPE: (
            "23gabonx0402",
            [
                "23gabonx0403",
                "23gabonx0404",
                "23gabonx0405",
                "23gabonx0406",
                "23gabonx0407",
                "23gabonx0408",
                "23gabonx0409",
                "23gabonx0410",
                "23gabonx0411",
                "23gabonx0412",
                "23gabonx0413",
                "23gabonx0414",
            ],
        ),
        sites.MABOUNIE: (
            "23gabonx0802",
            [
                "23gabonx0803",
                "23gabonx0804",
                "23gabonx0805",
                "23gabonx0806",
                "23gabonx0807",
                "23gabonx0808",
                "23gabonx0809",
                "23gabonx0810",
                "23gabonx0811",
                "23gabonx0812",
                "23gabonx0813",
            ],
        ),
        sites.MITZIC: (
            "23gabonx0302",
            [
                "23gabonx0303",
                "23gabonx0304",
                "23gabonx0305",
                "23gabonx0306",
                "23gabonx0307",
                "23gabonx0308",
                "23gabonx0309",
                "23gabonx0310",
                "23gabonx0311",
                "23gabonx0312",
                "23gabonx0313",
            ],
        ),
        sites.MONDAH: (
            "23gabonx0504",
            [
                "23gabonx0505",
                "23gabonx0506",
                "23gabonx0507",
                "23gabonx0508",
                "23gabonx0509",
                "23gabonx0510",
                "23gabonx0511",
                "23gabonx0512",
                "23gabonx0513",
                "23gabonx0514",
                "23gabonx0515",
            ],
        ),
        sites.PONGARA: (
            "23gabonx0606",
            [
                "23gabonx0602",
                "23gabonx0603",
                "23gabonx0604",
                "23gabonx0605",
                "23gabonx0607",
                "23gabonx0609",
                "23gabonx0610",
                "23gabonx0611",
                "23gabonx0612",
            ],
        ),
    }[site]


def main_baseline_selection(site, band):
    primary, secondary_list = pass_lists_by_site(site)
    ds = open_dataset.open_dataset(site)
    for pass_b in secondary_list:
        pass_ae = primary.encode("ascii")
        pass_be = pass_b.encode("ascii")
        kz = ds[f"kz_{band.lower()}"].sel(pass_a=pass_ae, pass_b=pass_be).values
        mean_kz = np.nanmean(kz)
        h_bl, v_bl = metadata.get_spatial_baselines_between(ds, primary, pass_b, band)

        target_h_bl = 15 if band == "L" else 30
        target_v_bl = 20 if band == "L" else 40
        fsar_bl = h_bl if site == sites.PONGARA else v_bl
        target_bl = target_h_bl if site == sites.PONGARA else target_v_bl

        if np.abs(np.abs(fsar_bl) - np.abs(target_bl)) < 2:
            print(f"{site} {band} {primary} {pass_b}   mean_kz {mean_kz:6.3f}   h_bl {h_bl:5.1f}   v_bl {v_bl:5.1f}")


if __name__ == "__main__":
    for site in [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]:
        for band in ["L", "P"]:
            main_baseline_selection(site, band)
