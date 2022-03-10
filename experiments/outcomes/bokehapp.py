''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve bokehapp.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import os
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

import panel as pn
import hvplot.pandas  # noqa

from bokeh.sampledata.iris import flowers

x = pn.widgets.Select(name='x', options=['sepal_width', 'petal_width'])
y = pn.widgets.Select(name='y', options=['sepal_length', 'petal_length'])
kind = pn.widgets.Select(name='kind', value='scatter', options=['bivariate', 'scatter'])
by_species = pn.widgets.Checkbox(name='By species')
color = pn.widgets.ColorPicker(value='#ff0000')

@pn.depends(by_species, color)
def by_species_fn(by_species, color):
    return 'species' if by_species else color

plot = flowers.hvplot(x=x, y=y, kind=kind, c=by_species_fn, colorbar=False, width=600, legend='top_right')

pn.Row(pn.WidgetBox(x, y, kind, color, by_species), plot)