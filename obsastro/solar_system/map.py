import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from astropy import time as apt, coordinates as apc
from astropy.table import Table
from bs4 import BeautifulSoup
from cartopy import crs as ccrs
from matplotlib import pyplot as plt, path as mpath, collections as mcol, transforms as mtransforms
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from tqdm import tqdm

from obsastro.api_provider import ApiService
from obsastro.cache.cache import AstroCache
from obsastro.solar_system.horizons import  HorizonsInterface
from obsastro.utils.general import figaxes

from obsastro.utils.projection import new_x_axis_at, unit_vector, current_x_axis_to


TwoValues = tuple[float, float]
logger = logging.getLogger(__name__)
usgs_map_cache = AstroCache(max_cache=int(1e12), lifetime=30,
                            hashable_kw=['detail'], label_on_disk='USGSmap',
                            force= False,
                            # verbose=True,
                            )



class BodyGeometry:
    """
    Keeps geometry of a body at a single time with info as returned from JPL Horizons.

    Attributes
    ----------
    sub_obs : tuple(float, float)
       Longitude and latitude in body of sub observer point.
    sub_obs_np : float
       angle (East of North) for the North Pole of the object as seen from sub-observer point.
    sub_sun : tuple(float, float)
       Longitude and latitude in body of sub solar point.
    ang_diam : float
       Angular diameter of the body as seen from the observer.
    """
    def __init__(self,
                 ephemeris):
        """
        Initialize the BodyGeometry with ephemeris data.

        Parameters
        ----------
        ephemeris : astropy.table.Row
            Row from an astropy Table containing ephemeris data from JPL Horizons
        """
        self._ephemeris = ephemeris
        self.sub_obs = ephemeris['ObsSub_LON'], ephemeris['ObsSub_LAT']
        self.sub_obs_np = ephemeris['NP_ang']
        self.sub_sun = ephemeris['SunSub_LON'], ephemeris['SunSub_LAT'],
        self.ang_diam = ephemeris['Ang_diam']

        self._rotate_to_subobs = new_x_axis_at(*self.sub_obs, z_pole_angle=-self.sub_obs_np)
        self._rotate_to_subsol = new_x_axis_at(*self.sub_sun)

    def print(self):
        """
        Print key geometry information about the body.

        Displays sub-observer and sub-solar points, as well as relative angles
        and distances.
        """
        print(f"Sub Observer longitude/latitude: {self._ephemeris['ObsSub_LON']}/{self._ephemeris['ObsSub_LAT']}")
        print(f"Sub Solar longitude/latitude: {self._ephemeris['SunSub_LON']}/{self._ephemeris['SunSub_LAT']}")
        print(f"Sub Solar angle/distances w/r to sub-observer: "
              f"{self._ephemeris['SN_ang']}/{self._ephemeris['SN_dist']}")
        print(f"North Pole angle/distances w/r to sub-observer: "
              f"{self._ephemeris['NP_ang']}/{self._ephemeris['NP_dist']}")

    def location(self, lon, lat,
                 label: str | None = None,
                 max_title_len: int = 0,
                 ):
        """
        Calculate the apparent position of a surface location as seen from the observer.

        Parameters
        ----------
        lon : float
            Longitude of the location on the body (degrees)
        lat : float
            Latitude of the location on the body (degrees)
        label : str, optional
            Name of the location for display purposes
        max_title_len : int, optional
            Maximum length of the label for formatting

        Returns
        -------
        dict
            Dictionary containing various geometric parameters for the location:
            'delta_ra' : float
                Offset in right ascension from body center (arcsec)
            'delta_dec' : float
                Offset in declination from body center (arcsec)
            'unit_ra' : float
                Normalized coordinate on RA axis
            'unit_dec' : float
                Normalized coordinate on Dec axis
            'local_time' : float
                Local solar time at the location (hours)
            'incoming' : float
                Solar incidence angle (degrees)
            'outgoing' : float
                Emission angle (degrees)
            'visible' : bool
                Whether the location is on the visible hemisphere
        """
        position = unit_vector(lon, lat, degrees=True)

        unit_with_obs_x = self._rotate_to_subobs.apply(position)
        unit_with_sun_x = self._rotate_to_subsol.apply(position)

        delta_ra, delta_dec = self.ang_diam * unit_with_obs_x[1:] / 2
        local_time_location = (np.arctan2(*unit_with_sun_x[:2][::-1]) + np.pi) * 12 / np.pi
        incoming_angle = np.arccos(unit_with_sun_x[0]) * 180 / np.pi
        emission_angle = np.arccos(unit_with_obs_x[0]) * 180 / np.pi

        if label is not None:
            format_str = [f"{{name:{max_title_len + 1}s}} {{delta_ra:+10.2f}} {{delta_dec:+10.2f}}",
                          f"   (LocalSolarTime: {{local_time_location:+06.2f}}h, ",
                          f"inc/emi ang: {{incoming_angle:.0f}}/{{emission_angle:.0f}}deg)"]
            print("".join(format_str).format(name=str(label),
                                             local_time_location=local_time_location,
                                             delta_ra=delta_ra,
                                             delta_dec=delta_dec,
                                             emission_angle=emission_angle,
                                             incoming_angle=incoming_angle))
        return {'delta_ra': delta_ra,
                'delta_dec': delta_dec,
                'unit_ra': unit_with_obs_x[1],
                'unit_dec': unit_with_obs_x[2],
                'local_time': local_time_location,
                'incoming': incoming_angle,
                'outgoing': emission_angle,
                'visible': unit_with_obs_x[0] > 0,
                }


class BodyVisualizer:
    """
    Class to visualize solar system bodies.

    This class provides static methods to create images and animations of solar system
    bodies using orthographic projections and JPL ephemeris data.
    """

    @staticmethod
    def get_orthographic(platecarree_image,
                     sub_obs_lon,
                     sub_obs_lat,
                     marks=None,
                     show_poles="",
                     ):
        """
        Returns orthographic projection with the specified center.

        Parameters
        ----------
        platecarree_image : PIL.Image
            Image of the body in plate carree (equirectangular) projection
        sub_obs_lon : float
            Longitude of the sub-observer point (degrees)
        sub_obs_lat : float
            Latitude of the sub-observer point (degrees)
        marks : list, optional
            List of (lon, lat) tuples marking positions to highlight
        show_poles : str, optional
            Color to mark poles, set to "" to skip

        Returns
        -------
        PIL.Image
            Image showing orthographic projection of the body as seen from the
            specified sub-observer point
        """
        projection = ccrs.Orthographic(sub_obs_lon,
                                    sub_obs_lat)

        backend = plt.get_backend()
        plt.switch_backend('agg')

        f = plt.figure(figsize=(3, 3))
        tmp_ax = f.add_axes((0, 0, 1, 1),
                            transform=f.transFigure,
                            projection=projection)

        tmp_ax.imshow(platecarree_image,
                    origin='upper',
                    transform=ccrs.PlateCarree(),
                    extent=(-180, 180, -90, 90),
                    )
        if show_poles:
            tmp_ax.plot(0, 90,
                        transform=ccrs.PlateCarree(),
                        marker='d', color=show_poles)
            tmp_ax.plot(0, -90,
                        transform=ccrs.PlateCarree(),
                        marker='d', color=show_poles)

        if marks is not None:
            for mark in marks:
                tmp_ax.plot(*mark,
                            transform=ccrs.PlateCarree(),
                            marker='x', color=show_poles
                            )

        tmp_ax.axis('off')
        tmp_ax.set_global()  # the whole globe limits
        f.canvas.draw()

        image_flat = np.frombuffer(f.canvas.tostring_argb(), dtype='uint8')  # (H * W * 3,)

        orthographic_image = image_flat.reshape(*reversed(f.canvas.get_width_height()), 4)[:, :, 1:]
        orthographic_image = Image.fromarray(orthographic_image, 'RGB')
        plt.close(f)

        plt.switch_backend(backend)

        return orthographic_image
    @staticmethod
    @usgs_map_cache
    def usgs_map_image(body,  warning_shown, detail=None, warn_multiple=True,verbose =False, force = False):
        """
        Retrieve a map image for a solar system body from USGS.

        Parameters
        ----------
        body : str
            Name of the body
        detail : str, optional
            Space-separated keywords to filter map alternatives
        warn_multiple : bool, default=True
            Whether to show warnings when multiple maps are available
        verbose : bool, default=False
            Whether to print verbose output during the fetching process
        force : bool, default=False
            If it is True, it will force fetching the map from USGS even if a cached version exists or the cache is configured to not force computation.

        Returns
        -------
        PIL.Image or None
            Image of the body map if successful, None otherwise

        Notes
        -----
        This method is decorated with usgs_map_cache to cache results and avoid
        redundant requests to the USGS server.
        """
        if verbose:
            logger.info(f"Fetching USGS map for {body} with detail '{detail}'")
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        def _parse_date(string):
            if string is None:
                return ""
            match len(string):
                case 8:
                    return f"{string[:4]}-{month[int(string[4:6])-1]}{string[6:8]}"
                case 10:
                    return f"{string[:4]}-{month[int(string[5:7]) - 1]}{string[8:10]}"

            raise NotImplementedError(f"Needs to implement parsing for date: {string}")

        directory = (Path(__file__).parents[0] / 'images')
        files = list(directory.glob("*.xml"))

        keywords = None
        if detail is not None:
            keywords = detail.split()

        # filter alternatives
        body_files = []
        for file in files:
            with open(file, 'r', encoding='utf8') as f:
                data = BeautifulSoup(f.read(), 'xml')
                body_in_xml = data.find("target")
                body_in_file = body_in_xml.string if body_in_xml is not None else file.name.split("_")[0]

                if body.lower() == body_in_file.lower():
                    title = data.idinfo("title")[0].string
                    if keywords is not None and not [k for k in keywords if k in title]:
                        continue

                    info = [title,
                            file,
                            data.find("browsen").string,
                            data.idinfo("begdate")[0].string,
                            data.idinfo("enddate")[0].string,
                            ]
                    if 'default' in str(file):
                        body_files.insert(0, info)
                    else:
                        body_files.append(info)

        detail_str = f" with detail keywords '{detail}'"
        if len(body_files) == 0:
            logger.error(f"No map of '{body}' found{detail_str}")
            raise ValueError(f"No map of '{body}' found{detail_str}")
        # select from alternatives
        elif len(body_files) > 1:
            if warn_multiple and not warning_shown:
                suggest = f" (use space-separated keywords in 'detail' to filter).\n" if not detail_str else ""
                print(f"Several map alternatives for {body} were available{detail_str}\n{suggest}"                  
                    "Selected first of:")
                for bf in body_files:
                    print(f"* {bf[0]} [{_parse_date(bf[3])}..{_parse_date(bf[4])}]")
                print("")

            body_files = [body_files[0]]
        if verbose:
            logger.info(f"Fetching map for {body}...")
        apiService = ApiService(verbose=verbose)
        response = apiService.request_http(url= body_files[0][2],
                                        timeout=10,
                                        method = "GET"
                                        )
        if response.success:

            return Image.open(BytesIO(response.data))
        else:

            return None
    @staticmethod
    def _cross2(a: np.ndarray,
            b: np.ndarray) -> np.ndarray:
        """
        Compute the cross product of two vectors.

        Parameters
        ----------
        a : numpy.ndarray
            First vector
        b : numpy.ndarray
            Second vector

        Returns
        -------
        numpy.ndarray
            Cross product of a and b

        Notes
        -----
        This is a workaround as np.cross does not return according to code inspection.
        """
        return np.cross(a,b)

    @staticmethod
    def _add_phase_shadow(ax,
                      sub_obs,
                      sub_sun,
                      np_ang,
                      color,
                      transform_from_normal,
                      precision=50,
                      marker_color='yellow',
                      ):
        """
        Add night side shading and sub-solar point marker to the visualization.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on
        sub_obs : tuple(float, float)
            Sub-observer point (longitude, latitude) in degrees
        sub_sun : tuple(float, float)
            Sub-solar point (longitude, latitude) in degrees
        np_ang : float
            North pole angle in degrees
        color : str
            Color for the shadow
        transform_from_normal : matplotlib.transforms.Transform
            Transform from normalized coordinates to data coordinates
        precision : int, default=50
            Number of points to use for the terminator curve
        marker_color : str, default='yellow'
            Color for the sub-solar point marker

        Returns
        -------
        tuple
            (PathCollection, Line2D) - shadow area and sub-solar point marker
        """
        rotate_body_to_subobs = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
        rotate_body_to_subsol = new_x_axis_at(*sub_sun)

        upper_vector_terminator = BodyVisualizer._cross2(unit_vector(*sub_obs), unit_vector(*sub_sun))
        upper_shadow_from_sun = new_x_axis_at(*sub_sun).apply(upper_vector_terminator)
        upper_angle_from_sun = (np.arctan2(*upper_shadow_from_sun[1:][::-1]) - np.pi / 2) * 180 / np.pi

        starting_terminator = new_x_axis_at(0, 90).apply(unit_vector(np.linspace(0, 360, precision), 0))
        terminator_at_body = current_x_axis_to(*sub_sun, z_pole_angle=-upper_angle_from_sun).apply(starting_terminator)
        terminator_sub_obs = np.array([(y, z) for x, y, z in rotate_body_to_subobs.apply(terminator_at_body) if x > 0])

        delta = ((terminator_sub_obs[1:,0] - terminator_sub_obs[:-1,0])**2 +
                (terminator_sub_obs[1:,1] - terminator_sub_obs[:-1,1])**2)
        max_delta = np.argmax(delta)
        if delta[max_delta] > 4/precision:
            terminator_sub_obs = np.roll(terminator_sub_obs, -(max_delta+1), axis=0)

        angle_first, angle_last = np.arctan2(*np.array(terminator_sub_obs)[[0, -1]].transpose()[::-1])

        if angle_first > angle_last:
            angle_last += 2*np.pi

        angle_perimeter = np.linspace(angle_last, angle_first, precision)
        perimeter = np.array([np.array([np.cos(ang), np.sin(ang)])
                            for ang in angle_perimeter]
                            + list(terminator_sub_obs))

        clip_path = mpath.Path(vertices=perimeter, closed=True)

        col = mcol.PathCollection([clip_path],
                                facecolors=color, alpha=0.7,
                                edgecolors='none',
                                zorder=7,
                                transform=transform_from_normal,
                                )
        ax.add_collection(col)

        projected_sub_sun = rotate_body_to_subobs.apply(unit_vector(*sub_sun))
        sub_sun_marker = ax.plot(*projected_sub_sun[1:],
                                marker='d', color=marker_color,
                                alpha=1 - 0.5 * (projected_sub_sun[0] < 0),
                                transform=transform_from_normal)
        ax.annotate(f"{np.abs(sub_sun[1]):.1f}$^\\circ${'N' if sub_sun[1] > 0 else 'S'}",
                    projected_sub_sun[1:],
                    xycoords=transform_from_normal,
                    color=marker_color,
                    )

        return col, sub_sun_marker

    @staticmethod
    def _add_local_time(ax,
                    sub_obs,
                    sub_sun,
                    np_ang,
                    color,
                    transform_from_norm,
                    precision=50,
                    ):
        """"Adds every hour of body's local time as labeled iso-longitude lines"""
        lunar_to_observer = new_x_axis_at(*sub_obs, z_pole_angle=-np_ang)
        local_time_to_body = current_x_axis_to(*sub_sun)

        artists = []
        for ltime in range(24):
            longitude_as_local_time = new_x_axis_at((12-ltime)*15, 0)

            local_time_rotation = lunar_to_observer * local_time_to_body * longitude_as_local_time
            local_time = local_time_rotation.apply(unit_vector(0, np.linspace(-90, 90, precision),
                                                            degrees=True)
                                                )
            # do not plot arcs that are in the hidden side of the object
            visible = np.array([(y, z) for x, y, z in local_time if x > 0])
            n_visible = len(visible)
            if n_visible > precision//2:  # if more than half the arc is visible then plot it.
                lines = ax.plot(visible[:, 0], visible[:, 1],
                                color=color,
                                alpha=0.7,
                                ls=':',
                                transform=transform_from_norm,
                                zorder=6,
                                linewidth=0.5,
                                )

                text = ax.annotate(f"{ltime}$^h$", (visible[n_visible // 2][0],
                                                    visible[n_visible // 2][1]),
                                color=color,
                                xycoords=transform_from_norm,
                                alpha=0.7, ha='center', va='center',
                                )

                artists.extend([lines, text])

        return tuple(artists)

    @staticmethod
    def create_frame( body, ephemeris_line, locations=None, title=None,
                detail=None, reread_usgs=False, radius_to_plot=None,
                ax=None, return_axes=True, verbose=False,
                color_location="red", color_phase='black',
                color_background='black', color_title='white',
                color_local_time='black', color_poles="blue",
                show_angles=False):
        """
        Generate a single frame visualization of a solar system body at a given time.

        Parameters
        ----------
        body : str
            Name of the body to visualize
        ephemeris_line : astropy.table.Row
            Row from an ephemeris table containing viewing geometry data
        locations : list or dict, optional
            Locations on the body to mark, either as list of (lon, lat) tuples
            or dict with name: (lon, lat) entries
        title : str, optional
            Title for the visualization (auto-generated if None)
        detail : str, optional
            Space-separated keywords to filter map alternatives
        reread_usgs : bool, default=False
            Whether to ignore the cache and fetch fresh map data
        radius_to_plot : float, optional
            Radius of the field of view in arcseconds (auto-detected if None)
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (created if None)
        return_axes : bool, default=True
            If True, return the Axes object; if False, return list of artists
        verbose : bool, default=False
            Whether to print detailed information
        color_location : str, default="red"
            Color for location markers and labels
        color_phase : str, default='black'
            Color for night side shading and local time indicators
        color_background : str, default='black'
            Background color for the figure
        color_title : str, default='white'
            Color for the title text
        color_local_time : str, default='black'
            Color for local time lines and labels
        color_poles : str, default="blue"
            Color for pole markers and coordinate lines
        show_angles : bool, default=False
            Whether to show incidence and emission angles in location labels
        first_time_warned : bool ,  default = False
            Whether if the user was warned about multiple usgs data files

        Returns
        -------
        matplotlib.axes.Axes or list
            Axes object or list of artists depending on return_axes

        Raises
        ------
        ValueError
            If the map image for the body cannot be retrieved
        """
        geometry = BodyGeometry(ephemeris_line)
        if verbose:
            geometry.print()

        if not hasattr(BodyVisualizer.create_frame, "_warning_shown"):
            BodyVisualizer.create_frame._warning_shown = False


        image = BodyVisualizer.usgs_map_image(body,
                                            detail=detail,
                                            warn_multiple=not reread_usgs,
                                            warning_shown=BodyVisualizer.create_frame._warning_shown,
                                            verbose =verbose,
                                            force= reread_usgs)
        if image is None:
            raise ValueError(f"Could not get image for {body}")

        if not BodyVisualizer.create_frame._warning_shown:
            BodyVisualizer.create_frame._warning_shown = True

        orthographic_image = BodyVisualizer.get_orthographic(image, *geometry.sub_obs,
                                          show_poles=color_poles)
        rotated_image = orthographic_image.rotate(geometry.sub_obs_np,
                                              resample=Image.Resampling.BICUBIC,
                                              expand=False, fillcolor=(255, 255, 255))
        x_offset = 0.5
        y_offset = 0
        ny, nx = rotated_image.size
        yy, xx = np.mgrid[-ny/2 + y_offset: ny/2 + y_offset, -nx/2 + x_offset: nx/2 + x_offset]
        rr = np.sqrt(yy**2 + xx**2).flatten()

        color_background_rgb = (*[int(c*255) for c in to_rgba(color_background)[:3]], 0)
        rotated_image.putdata([item if r < nx/2 - 1 else color_background_rgb
                            for r, item in zip(rr, rotated_image.convert("RGBA").getdata())])

        f, ax = figaxes(ax)
        f.patch.set_facecolor(color_background)
        ax.set_facecolor(color_background)
        # ax.imshow(rotated_image,
        #         )
        # ax.axis('off')

        ax.set_facecolor(color_background)
        ang_rad = geometry.ang_diam/2
        ax.imshow(rotated_image,
                extent=[-ang_rad, ang_rad, -ang_rad, ang_rad],
                )

        if radius_to_plot is None:
            radius_to_plot = ang_rad

        ax.set_xlim([-radius_to_plot, radius_to_plot])
        ax.set_ylim([-radius_to_plot, radius_to_plot])


        time = apt.Time(ephemeris_line['jd'], format='jd')
        if color_title is not None and color_title:
            if title is None:
                radius = ang_rad
                rad_unit = '"'
                if radius > 120:
                    radius /= 60
                    rad_unit = "'"
                title = f'{body.capitalize()} on {time.isot[:16]} (R$_{body[0].upper()}$: {radius:.1f}{rad_unit})'
            ax.set_title(title,
                        color=color_title,
                        )
        transform_norm_to_axes = mtransforms.Affine2D().scale(ang_rad) + ax.transData

        if locations is not None and len(locations) > 0:
            if verbose:
                print(f"Location offset from {body.capitalize()}'s center (Delta_RA, Delta_Dec) in arcsec:")

            if isinstance(locations, list):
                locations = {str(i): v for i, v in enumerate(locations)}
            max_len = max([len(str(k)) for k in locations.keys()])
            for name, location in locations.items():
                location = geometry.location(*location, label=name if verbose else None,
                                            max_title_len=max_len)

                incoming_emission = f"{location['incoming']:.0f}/{location['outgoing']:.0f}"
                ie_plot = f", i/e {incoming_emission}$^{{\\circ}}$" if show_angles else ""

                ax.plot(location['unit_ra'], location['unit_dec'],
                        transform=transform_norm_to_axes,
                        marker='d', color=color_location,
                        alpha=1 - 0.5 * location['visible'],
                        zorder=10,
                        )

                ax.annotate(f"{str(name)}: $\\Delta\\alpha$ {location['delta_ra']:+.0f}\", "
                            f"$\\Delta\\delta$ {location['delta_dec']:.0f}\", "
                            f"LT{location['local_time']:04.1f}$^h${ie_plot}",
                            (location['unit_ra'], location['unit_dec']),
                            xycoords=transform_norm_to_axes,
                            color=color_location,
                            alpha=1 - 0.5 * location['visible'],
                            zorder=10,
                            )

            if verbose:
                print("")

        if color_poles is not None and color_poles:
            ax.plot([-1, 1], [0, 0],
                    color='blue',
                    transform=transform_norm_to_axes,
                    zorder=9,
                    )
            ax.plot([0, 0], [-1, 1],
                    color='blue',
                    transform=transform_norm_to_axes,
                    zorder=9,
                    )

            for lat_pole in [-90, 90]:
                pole = geometry.location(0, lat_pole)
                ax.plot(pole['unit_ra'], pole['unit_dec'],
                        transform=transform_norm_to_axes,
                        alpha=1 - 0.5*pole['visible'],
                        marker='o', color='blue',
                        zorder=9,
                        )

        if color_local_time:
            BodyVisualizer._add_local_time(ax,
                            geometry.sub_obs,
                            geometry.sub_sun,
                            geometry.sub_obs_np,
                            color_phase,
                            transform_norm_to_axes,
                            )

        if color_phase is not None and color_phase:
            BodyVisualizer._add_phase_shadow(ax,
                            geometry.sub_obs,
                            geometry.sub_sun,
                            geometry.sub_obs_np,
                            color_phase,
                            transform_norm_to_axes,
                            )

        if return_axes:
            return ax
        else:
            return ax.collections + ax.lines + ax.texts + ax.images

    @staticmethod
    def create_image(body, ephemeris_line, locations=None, title=None,
               detail=None, reread_usgs=False, radius_to_plot=None,
               ax=None, return_axes=True, verbose=False,
               color_location="red", color_phase='black',
               color_background='black', color_title='white',
               color_local_time='black', color_poles="blue",
               show_angles=False):
        """
        Create an image of a solar system body.

        This is a wrapper around create_frame that emphasizes its use for
        static image creation.

        Parameters
        ----------
        body : str
            Name of the body to visualize
        ephemeris_line : astropy.table.Row
            Row from an ephemeris table containing viewing geometry data
        locations : list or dict, optional
            Locations on the body to mark, either as list of (lon, lat) tuples
            or dict with name: (lon, lat) entries
        title : str, optional
            Title for the visualization (auto-generated if None)
        detail : str, optional
            Space-separated keywords to filter map alternatives
        reread_usgs : bool, default=False
            Whether to ignore the cache and fetch fresh map data
        radius_to_plot : float, optional
            Radius of the field of view in arcseconds (auto-detected if None)
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (created if None)
        return_axes : bool, default=True
            If True, return the Axes object; if False, return list of artists
        verbose : bool, default=False
            Whether to print detailed information
        color_location : str, default="red"
            Color for location markers and labels
        color_phase : str, default='black'
            Color for night side shading and local time indicators
        color_background : str, default='black'
            Background color for the figure
        color_title : str, default='white'
            Color for the title text
        color_local_time : str, default='black'
            Color for local time lines and labels
        color_poles : str, default="blue"
            Color for pole markers and coordinate lines
        show_angles : bool, default=False
            Whether to show incidence and emission angles in location labels

        Returns
        -------
        matplotlib.axes.Axes or list
            Axes object or list of artists depending on return_axes
        """
        return BodyVisualizer.create_frame(
            body=body,
            ephemeris_line=ephemeris_line,
            locations=locations,
            title=title,
            detail=detail,
            reread_usgs=reread_usgs,
            radius_to_plot=radius_to_plot,
            ax=ax,
            return_axes=return_axes,
            verbose=verbose,
            color_location=color_location,
            color_phase=color_phase,
            color_background=color_background,
            color_title=color_title,
            color_local_time=color_local_time,
            color_poles=color_poles,
            show_angles=show_angles
            )

    @staticmethod
    def create_video(body, ephemeris_lines, times, filename=None, fps=10, dpi=75,
                  locations=None, title=None, detail=None, reread_usgs=False,
                  radius_to_plot=None, ax=None, verbose=False,
                  color_location="red", color_phase='black',
                  color_background='black', color_title='white',
                  color_local_time='black', color_poles="blue",
                  show_angles=False):
        """
        Create an animated visualization of a solar system body over time.

        Parameters
        ----------
        body : str
            Name of the body to visualize
        ephemeris_lines : astropy.table.Table
            Table containing ephemeris data for each frame
        times : astropy.time.Time
            Array of times corresponding to each frame
        filename : str, optional
            Output filename (auto-generated if None)
        fps : int, default=10
            Frames per second in the output animation
        dpi : int, default=75
            Resolution of the output animation
        locations : list or dict, optional
            Locations on the body to mark, either as list of (lon, lat) tuples
            or dict with name: (lon, lat) entries
        title : str, optional
            Title template for the visualization (can use {time} and {field} placeholders)
        detail : str, optional
            Space-separated keywords to filter map alternatives
        reread_usgs : bool, default=False
            Whether to ignore the cache and fetch fresh map data
        radius_to_plot : float, optional
            Radius of the field of view in arcseconds (auto-detected if None)
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (created if None)
        verbose : bool, default=False
            Whether to print detailed information
        color_location : str, default="red"
            Color for location markers and labels
        color_phase : str, default='black'
            Color for night side shading and local time indicators
        color_background : str, default='black'
            Background color for the figure
        color_title : str, default='white'
            Color for the title text
        color_local_time : str, default='black'
            Color for local time lines and labels
        color_poles : str, default="blue"
            Color for pole markers and coordinate lines
        show_angles : bool, default=False
            Whether to show incidence and emission angles in location labels

        Returns
        -------
        str or None
            Path to the saved animation file, or None if there was an error
        """
        # Save the current backend and switch to agg
        backend = plt.get_backend()
        plt.switch_backend('agg')

        # Set up the figure
        f, ax = figaxes()

        f.patch.set_facecolor(color_background)
        ax.set_facecolor(color_background)

        # Calculate radius to plot if not provided
        field = 'Ang_diam'
        if radius_to_plot is None:
            radius_to_plot = np.max(ephemeris_lines[field])/2

        # Generate default title if not provided
        if title is None:
            mean_rad = np.mean(ephemeris_lines[field])
            if mean_rad > 120:
                rad_unit = "'"
                divider = 60
            elif mean_rad > 1:
                rad_unit = '"'
                divider = 1
            else:
                rad_unit = 'mas'
                divider = 0.001
            title = f'{body.capitalize()} on {{time}} (R$_{body[0].upper()}$: {{field:.1f}}{rad_unit})'

        # Global progress counter
        frame_count = [0]
        pbar = tqdm(total=len(times), desc=f"Animating {len(times)} frames")
        def animate(itime):
            # Clear current axes
            ax.clear()

            # Safety check to prevent extra iterations
            if itime >= len(times):
                return []

            # Get ephemeris data for the current time
            ephemeris = ephemeris_lines[itime]

            # Format the time label
            y, m, d, h, mn, s = times[itime].ymdhms
            time_label = f"{y}.{m:02d}.{d:02d} {h:02d}:{mn+s/60:04.1f} UT"

            # Generate the frame
            artists = BodyVisualizer.create_frame(
                body, ephemeris,
                ax=ax,
                radius_to_plot=radius_to_plot,
                title=title.format(time=time_label, field=ephemeris[field]/divider),
                color_background=color_background,
                return_axes=False,  # Return artists instead of axes
                verbose=verbose,
                locations=locations,
                detail=detail,
                reread_usgs=reread_usgs,
                color_location=color_location,
                color_phase=color_phase,
                color_title=color_title,
                color_local_time=color_local_time,
                color_poles=color_poles,
                show_angles=show_angles,
            )



            # Update progress bar (but only once per frame)
            if frame_count[0] < len(times):
                pbar.update(1)
                frame_count[0] += 1

            return artists

        # Create the animation with exact number of frames
        ani = FuncAnimation(
            f,
            animate,
            frames=len(times),  # Explicitly set number of frames
            blit=True,  # Use blitting for better performance
            repeat=False,  # Don't repeat the animation
            interval=1000/fps  # Set frame interval based on fps
        )

        # Set up the filename
        if filename is None:
            time_str = times[0].isot.replace(':', '')
            if len(time_str) > 17:
                time_str = time_str[:17]
            filename = f"{body}_{time_str}.gif"
        elif '.' not in filename:
            filename += '.gif'
        if verbose:
            logger.info(f"Creating animation with {len(times)} frames...")

        # Select the appropriate writer
        if filename.lower().endswith(('.mpg', '.mpeg', '.mp4')):
            writer = FFMpegWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)

        # Save the animation
        try:
            ani.save(filename, dpi=dpi, writer=writer)
            pbar.close()
            logger.info(f"Saved file {filename}")
        except Exception as e:
            pbar.close()
            logger.error(f"Error saving animation: {e}")
            plt.switch_backend(backend)
            return None

        plt.switch_backend(backend)
        return filename


def body_map(body:str,
             observer:str,
             time: apt.Time | None = None,
             locations: list[TwoValues] | dict[str, TwoValues] = None,
             title:str=None,
             detail:str=None, reread_usgs=False,
             radius_to_plot=None,
             fps=10, dpi=75, filename=None,
             ax: Axes | None = None, return_axes=True, verbose=False,
             color_location="red", color_phase='black',
             color_background='black', color_title='white',
             color_local_time='black', color_poles="blue",
             show_angles=False,
             ):
    """
    Generate visualization of solar system bodies as either static images or animations.

    This is the main user-facing function that combines the functionality of
    BodyVisualizer.create_image and BodyVisualizer.create_video depending on
    whether time is a scalar or array.

    NOTE: The Verbose Parameter is expected to cause tqdm (progress bar) bugs and repeated rendering of the loading bar.

    Parameters
    ----------
    body : str
        Name of the body to visualize
    observer : str or astropy.table.Row
        Either location name or ephemeris row
    time : astropy.time.Time, optional
        Time(s) to visualize. If scalar, creates an image; if array, creates a video;
        if None, uses time from ephemeris in observer
    locations : list or dict, optional
        Locations on the body to mark, either as list of (lon, lat) tuples
        or dict with name: (lon, lat) entries
    title : str, optional
        Title for the visualization (auto-generated if None)
    detail : str, optional
        Space-separated keywords to filter map alternatives
    reread_usgs : bool, default=False
        Whether to ignore the cache and fetch fresh map data. This will override the directive of the cache creation if it is set to false its force attribute.
    radius_to_plot : float, optional
        Radius of the field of view in arcseconds (auto-detected if None)
    fps : int, default=10
        Frames per second for video output (ignored for static images)
    dpi : int, default=75
        Resolution for video output (ignored for static images)
    filename : str, optional
        Output filename for video (auto-generated if None, ignored for static images)
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (created if None)
    return_axes : bool, default=True
        If True, return the Axes object; if False, return list of artists
        (only applies to static images)
    verbose : bool, default=True
        Whether to print detailed information
    color_location : str, default="red"
        Color for location markers and labels
    color_phase : str, default='black'
        Color for night side shading and local time indicators
    color_background : str, default='black'
        Background color for the figure
    color_title : str, default='white'
        Color for the title text
    color_local_time : str, default='black'
        Color for local time lines and labels
    color_poles : str, default="blue"
        Color for pole markers and coordinate lines
    show_angles : bool, default=False
        Whether to show incidence and emission angles in location labels

    Returns
    -------
    matplotlib.axes.Axes, list, or str
        For static images, returns Axes object or list of artists depending on return_axes.
        For animations, returns the path to the saved file.

    Raises
    ------
    TypeError
        If time is None but observer is not an ephemeris row
    """
    # Common parameters for both image and video
    common_params = {
        'locations': locations,
        'title': title,
        'detail': detail,
        'reread_usgs': reread_usgs,
        'radius_to_plot': radius_to_plot,
        'ax': ax,
        'verbose': verbose,
        'color_location': color_location,
        'color_phase': color_phase,
        'color_background': color_background,
        'color_title': color_title,
        'color_local_time': color_local_time,
        'color_poles': color_poles,
        'show_angles': show_angles,
    }



    if time is None:
        if isinstance(observer, Table.Row):
            ephemeris_line = observer
            logger.info(f"Time is None. Using time from ephemeris: {apt.Time(ephemeris_line['jd'], format='jd')}")
            time = apt.Time(ephemeris_line['jd'], format='jd')
        else:
            logger.info("Time is None. Using now")
            time = apt.Time.now()

    # Case 1: time is either none or an scalar
    if time.isscalar:

        # Get ephemeris data for the specified time
        site = apc.EarthLocation.of_site(observer)
        request = HorizonsInterface.jpl_observer_from_location(site) | \
                HorizonsInterface.jpl_body_from_str(body) | \
                HorizonsInterface.jpl_times_from_time(time)
        ephemeris_line = HorizonsInterface.read_jpl(request)[0]

        return BodyVisualizer.create_image(
            body=body,
            ephemeris_line=ephemeris_line,
            return_axes=return_axes,
            **common_params
        )

    # Case 2: Time is an array
    else:

        site = apc.EarthLocation.of_site(observer)
        request = HorizonsInterface.jpl_observer_from_location(site) | \
                HorizonsInterface.jpl_body_from_str(body) | \
                HorizonsInterface.jpl_times_from_time(time)
        ephemeris_lines = HorizonsInterface.read_jpl(request)

        # Additional parameters for video
        video_params = {
            'filename': filename,
            'fps': fps,
            'dpi': dpi
        }

        return BodyVisualizer.create_video(
            body=body,
            ephemeris_lines=ephemeris_lines,
            times=time,
            **common_params,
            **video_params
        )
