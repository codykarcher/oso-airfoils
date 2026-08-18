from collections import namedtuple

ColorList = namedtuple('ColorList', ['blue', 'orange', 'green', 'red', 'purple', 'yellow', 'cyan', 'pink', 'gray', 'black'])

default_color_cycle = ColorList(
    '#0065cc', 
    '#eea800', 
    '#009e73', 
    '#d55e00', 
    '#7860aa', 
    '#ede13f', 
    '#56b4ff', 
    '#fca7c7', 
    '#5d5d5d', 
    '#000000'
)

# Per-family colors used consistently across comparison plots.
# Keys match the oso_airfoils/airfoils/ sub-directory names.
FAMILY_COLORS: dict[str, str] = {
    'du'          : default_color_cycle.blue,    # DU-series
    'ffa'         : default_color_cycle.orange,  # FFA-W series
    'riso_a'      : default_color_cycle.red,     # Risø-A
    'riso_b'      : default_color_cycle.green,   # Risø-B
    'riso_p'      : default_color_cycle.purple,  # Risø-P
    's'           : default_color_cycle.cyan,    # S-series (NREL)
    'mhkf1'       : default_color_cycle.pink,    # MHK-F1
    'oso_2025_wt1': default_color_cycle.black,
    'oso_2025_wt2': default_color_cycle.black,    # per-tau overrides used in plots
    'oso_2026_wt2s': default_color_cycle.black,
    'oso_2026_ht1': default_color_cycle.black,
    'oso_2026_wt3': default_color_cycle.black,
}

# # OSO WT2 airfoils are coloured by thickness tier.
# OSO_WT2_TAU_COLORS: dict[str, str] = {
#     '21': default_color_cycle.blue,
#     '24': default_color_cycle.orange,
#     '27': default_color_cycle.green,
#     '30': default_color_cycle.red,
#     '33': default_color_cycle.purple,
#     '36': default_color_cycle.cyan,
# }