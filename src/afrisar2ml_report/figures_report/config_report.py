from afrisar2ml_report import sites


def get_pass_pair(site):
    pass_a, pass_b = {
        sites.LOPE: ("23gabonx0402", "23gabonx0407"),
        sites.MABOUNIE: ("23gabonx0802", "23gabonx0808"),
        sites.MITZIC: ("23gabonx0302", "23gabonx0306"),
        sites.MONDAH: ("23gabonx0504", "23gabonx0508"),
        sites.PONGARA: ("23gabonx0606", "23gabonx0605"),
    }[site]
    return pass_a, pass_b
