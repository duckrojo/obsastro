import numpy as np
from matplotlib import pyplot as plt

from obsastro.solar_system.horizons import HorizonsInterface
import astropy.units as u
import astropy.time as apt
import astropy.coordinates as apc

site = apc.EarthLocation.of_site("paranal")
time = apt.Time("2026-07-26") + np.linspace(0, 4.5, 30*24*10)*30*u.day
request = HorizonsInterface.jpl_observer_from_location(site) | \
          HorizonsInterface.jpl_times_from_time(time)

ephemeris = HorizonsInterface.read_jpl(request | HorizonsInterface.jpl_body_from_str("io"))

one = apc.SkyCoord(ephemeris['ra_ICRF'], ephemeris['dec_ICRF'],unit= (u.hourangle, u.deg))

ephemeris = HorizonsInterface.read_jpl(request | HorizonsInterface.jpl_body_from_str("europa"))

two = apc.SkyCoord(ephemeris['ra_ICRF'], ephemeris['dec_ICRF'],unit= (u.hourangle, u.deg))

separation = one.separation(two)
daytime = [(hh>10  and hh<20) for yy,mm,dd,hh,mn,ss in time.ymdhms]
separation[np.array(daytime)] = np.nan

plt.plot(time.to_datetime(), separation.to(u.arcsec))
plt.ylim([0,20])
