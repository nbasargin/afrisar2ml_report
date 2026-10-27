def format_name(site_name):
    return site_name[0] + site_name[1:].lower()


def format_baseline(h_bl, v_bl):
    b_hor, b_ver = "B_{hor}", "B_{ver}"
    return f"${b_hor} = {h_bl:.0f}$, ${b_ver} = {v_bl:.0f}$"
