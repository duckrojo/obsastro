import matplotlib
import obsastro.exoplanet as oae

matplotlib.use('TkAgg')

print("Initializing...")
night = oae.Nightly()

print(" plotting...")
night.plot("2026-11-26", save_dir='out')

night.show()
# night.savefig("nightly.png")

