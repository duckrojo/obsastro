import matplotlib
import obsastro.exoplanet as oae

matplotlib.use('TkAgg')

print("Initializing...")
night = oae.Nightly()

print(" plotting...")
night.plot("2025-05-06")

night.show()
# night.savefig("nightly.png")

