import matplotlib

from obsastro.exoplanet.phases import ExoPlanet

matplotlib.use('TkAgg')

exo = ExoPlanet("wasp-76 b", "2025-05-01/2026-04-30", site="ctio",)

exo.plot_phases(shade=[0.17, 0.31], show=True)
exo.plot_phases(shade=[0.25, 0.375], show=True)
#exo.plot_phases(shade=[0.55, 0.69], show=True)