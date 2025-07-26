import matplotlib

from obsastro.exoplanet.phases import ExoPlanet

matplotlib.use('TkAgg')

ExoPlanet("wasp76 b", "2024-09-01/2024-12-01", site="ctio",
          ).plot_phases(shade=[0.25, 0.375], show=True)
