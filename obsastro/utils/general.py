#
#
# Copyright (C) 2013 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
#

__all__ = ['accept_object_name',
           ]

import numpy as np
import operator as op
import re

from matplotlib import pyplot as plt


def accept_object_name(name1, name2, planet_match=False, binary_match=False):
    """
    Check if two astronomical names are the same, case independently and punctuation independent.
    Binary stars are identified by upper case. Planets are identified from lower case b onwards.

    Parameters
    ----------
    binary_match: bool
        if True, then names must match binary identification (e.g. Gliese81A != gliese81B)
    planet_match: bool
        if True, then names must match planet identification (e.g. ProxCen b != ProxCen c)
    name1 : str
        Name of object1
    name2 : str
        Name of object1

    Returns
    -------
    bool
    """

    def name_items(name: str) -> tuple[str, str, str, str]:
        astro_name_re = re.compile(r"(?:(?:([a-zA-Z][a-zA-Z0-9]*?)-)|([a-zA-Z]+))[-_ ]?(\d*)([A-Z]*)[ _]?([b-z]*)")
        n_alternate_catalogs = 2  # how many catalog name version are matched (2: "Name|NameNumber-")

        items = astro_name_re.match(name).groups()
        catalog = None
        for n in range(n_alternate_catalogs):
            if catalog is None:
                catalog = items[n]

        return catalog, *items[n_alternate_catalogs:]

    catalog1, number1, binary1, planet1 = name_items(name1)
    catalog2, number2, binary2, planet2 = name_items(name2)

    if catalog1.lower() != catalog2.lower() or number1 != number2:
        return False
    if binary_match and binary1 != binary2:
        return False
    if planet_match and planet1 != planet2:
        return False
    return True


def figaxes(axes: int | plt.Figure | plt.Axes = None,
            force_new: bool = True,
            clear: bool = True,
            figsize: tuple[int, int] | None = None,
            nrows: int = 1,
            ncols: int = 1,
            projection=None,
            **kwargs,
            ) -> (plt.Figure, plt.Axes):
    """
    Function that accepts a variety of axes specifications  and returns the output
    ready for use with matplotlib

    Parameters
    ----------
    projection:
        projection to use if new axes need to be created. If using existing axes this parameter is omitted
    axes : int, plt.Figure, plt.Axes, None
        If axes is None, and multi col/row setup is requested, then it returns an array as in add_subplots().
        Otherwise, it always returns just one Axes instance.
    figsize: (int, int), optional
        Size of a figure, only valid for new figures (axes=None)
    force_new : bool, optional
        If true starts a new axe when axes=None (and only then) instead of using last figure
    clear: bool, optional
        Delete previous axes content, if any

    Returns
    -------
    Matplotlib.pyplot figure and axes
    """
    if axes is None:
        if force_new:
            fig, axs = plt.subplots(nrows, ncols,
                                    figsize=figsize,
                                    subplot_kw=dict(projection=projection),
                                    )
        else:
            plt.gcf().clf()
            fig, axs = plt.subplots(nrows, ncols,
                                    figsize=figsize,
                                    num=plt.gcf().number,
                                    subplot_kw=dict(projection=projection),
                                    )
    elif isinstance(axes, int):
        fig = plt.figure(axes, **kwargs)
        if clear or len(fig.axes) == 0:
            fig.clf()
            axs = fig.add_subplot(nrows, ncols, 1,
                                  projection=projection,
                                  )
        else:
            axs = fig.axes[0]
    elif isinstance(axes, plt.Figure):
        fig = axes
        if clear:
            fig.clf()
        if len(fig.axes) == 0:
            fig.add_subplot(nrows, ncols, 1,
                            projection=projection,
                            )
        axs = fig.axes[0]
    elif isinstance(axes, plt.Axes):
        axs = axes
        if clear:
            axs.cla()
        fig = axes.figure
    else:
        raise ValueError("Given value for axes ({0:s}) is not"
                         "recognized".format(axes, ))

    return fig, axs
