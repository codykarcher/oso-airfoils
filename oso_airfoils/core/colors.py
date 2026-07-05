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