import matplotlib

from procastro import obsrv

matplotlib.use('TkAgg')

obs = obsrv.Obsrv("WASP76b",site="ctio",timespan=2025, phase_offset=0.31
                  )
