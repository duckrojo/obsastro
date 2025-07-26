import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from numpy import ma

from astropy import time as apt, units as u, coordinates as apc, io as io
from astropy.table import QTable, MaskedColumn

from obsastro.api_provider.api_service import ApiService
from obsastro.cache.cache import AstroCache

jpl_cache = AstroCache(max_cache=int(1e12), lifetime=30,)


class HorizonsInterface:
    """
    Class to encapsulate the JPL Horizons interface.
    
    This class provides methods to interact with JPL's Horizons system to retrieve
    ephemeris data for solar system bodies. It handles request formatting, 
    submission, and parsing of the results.
    """
    
    @staticmethod
    @jpl_cache
    def _request_horizons_online(specifications):
        """
        Send request to JPL Horizons system and return the response.
        
        Parameters
        ----------
        specifications : str
            Horizons batch-style specifications
            
        Returns
        -------
        list
            List of strings representing the lines of the Horizons output
            
        Notes
        -----
        This method is decorated with jpl_cachev2 to cache results and avoid
        redundant requests to the JPL server.
        """
        default_spec = {'MAKE_EPHEM': 'YES',
                        'EPHEM_TYPE': 'OBSERVER',
                        'CENTER': "'500@399'",
                        'QUANTITIES': "'1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,"
                                    "27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48'",
                        'REF_SYSTEM': "'ICRF'",
                        'CAL_FORMAT': "'JD'",
                        'CAL_TYPE': "'M'",
                        'TIME_DIGITS': "'MINUTES'",
                        'ANG_FORMAT': "'HMS'",
                        'APPARENT': "'AIRLESS'",
                        'RANGE_UNITS': "'AU'",
                        'SUPPRESS_RANGE_RATE': "'NO'",
                        'SKIP_DAYLT': "'NO'",
                        'SOLAR_ELONG': "'0,180'",
                        'EXTRA_PREC': "'NO'",
                        'R_T_S_ONLY': "'NO'",
                        'CSV_FORMAT': "'NO'",
                        'OBJ_DATA': "'YES'",
                        }
        custom_spec = {}
        prev = ""
        for spec in specifications.split("\n"):
            if spec[:6] == r"!$$SOF":
                continue
            kv = spec.strip().split("=")
            if len(kv) == 2:
                custom_spec[kv[0]] = kv[1]
                prev = kv[0]
            else:
                custom_spec[prev] += " " + kv[0]

        url_api = "https://ssd.jpl.nasa.gov/api/horizons.api?"
        full_specs = [f"{k}={v}" for k, v in (default_spec | custom_spec).items()
                    if k != 'TLIST']

        url = url_api + "&".join(full_specs)
        if 'TLIST' in custom_spec:
            url += f'&TLIST={custom_spec["TLIST"]}'
        if len(url) > 1000:
            if 'TLIST' in custom_spec:
                epochs = custom_spec['TLIST'].split(' ')
                epochs[0] = 'TLIST=' + epochs[0]
                full_specs.extend(epochs)

            url_api_file = "https://ssd.jpl.nasa.gov/api/horizons_file.api?"
            with NamedTemporaryFile(mode="w", delete_on_close=False) as fp:
                fp.write("!$$SOF\n")
                fp.write("\n".join(full_specs))
                fp.close()

                # TODO: change this to the new api service model


                apiService = ApiService()
                response = apiService.request_http(url = url_api_file, method="POST", data={'format': 'text'}, files={'input': open(fp.name)})
                if not response.success:
                    raise ValueError(f"JPL Horizons request failed: {response.error_message}")
                data = response.data 
                return data.splitlines()


                
                # return requests.post(url_api_file,
                #                     data={'format': 'text'},
                #                     files={'input': open(fp.name)}
                #                     ).text.splitlines()
            
                

        else:
            apiService= ApiService(verbose=True) 
            result = apiService.request_http(url = url, allow_redirects=True)
            if not result.success:
                raise ValueError(f"JPL Horizons request failed: {result.error_message}")
            data = result.data
            return data['result'].splitlines()

    

    @staticmethod
    def get_jpl_ephemeris(specification):
        """
        Read JPL's Horizons ephemeris file returning the raw ephemeris data.
        
        Parameters
        ----------
        specification : str or dict
            Ephemeris specification. It can be the filename to be read with the ephemeris, 
            or with newline-separated commands for horizon's batch mode. 
            It can also be a newline-separated string with the specifications
            or a dictionary of key-value pairs.
            
        Returns
        -------
        list
            List of strings representing the raw ephemeris data
            
        Raises
        ------
        FileNotFoundError
            If a filename is provided and the file does not exist
        ValueError
            If the specification format is invalid
        """
        if isinstance(specification, dict):
            specification = f"""!$$SOF\n{"\n".join([f'{k}={v}' for k, v in specification.items()])}"""

        specification = specification.strip()
        if specification.count("\n") == 0:  # filename is given
            filename = Path(specification)
            if not filename.exists():
                raise FileNotFoundError(f"File '{filename}' does not exists")
            with open(filename, 'r') as fp:
                line = fp.readline()
                if line[:6] == r"!$$SOF":
                    ephemeris = HorizonsInterface._request_horizons_online(fp.read())
                else:
                    ephemeris = open(specification, 'r').readlines()
        else:
            if specification[:6] != r"!$$SOF":
                raise ValueError(f"Multiline Horizons specification invalid:"
                                f"{specification}")
            ephemeris = HorizonsInterface._request_horizons_online(specification)

        return ephemeris

    @staticmethod
    def parse_jpl_ephemeris(ephemeris):
        """
        Parse the raw JPL ephemeris data into an astropy Table.
        
        Parameters
        ----------
        ephemeris : list
            List of strings representing the raw ephemeris data
            
        Returns
        -------
        astropy.table.Table
            Table with named columns containing the parsed ephemeris data
            
        Raises
        ------
        ValueError
            If the ephemeris data is not properly formatted or missing required markers
        """
        ephemeris = ephemeris.copy()

        float_col = ['Date_________JDUT', 'APmag', 'S_brt',
                    'dRA_cosD', 'd_DEC__dt', 'dAZ_cosE', 'd_ELV__dt',
                    'SatPANG', 'a_mass', 'mag_ex',
                    'Illu_', 'Def_illu', 'Ang_diam',
                    'ObsSub_LON', 'ObsSub_LAT', 'SunSub_LON', 'SunSub_LAT',
                    'SN_ang', 'SN_dist', 'NP_ang', 'NP_dist', 'hEcl_Lon', 'hEcl_Lat',
                    'r', 'rdot', 'delta', 'deldot', 'one_way_down_LT', 'VmagSn', 'VmagOb',
                    'S_O_T', 'S_T_O', 'O_P_T', 'PsAng', 'PsAMV', 'PlAng',
                    'TDB_UT', 'ObsEcLon', 'ObsEcLat', 'N_Pole_RA', 'N_Pole_DC',
                    'GlxLon', 'GlxLat',  'Tru_Anom', 'phi',
                    'earth_ins_LT', 'RA_3sigma', 'DEC_3sigma', 'SMAA_3sig', 'SMIA_3sig', 'Theta Area_3sig',
                    'POS_3sigma', 'RNG_3sigma', 'RNGRT_3sig', 'DOP_S_3sig',  'DOP_X_3sig', 'RT_delay_3sig',
                    'PAB_LON', 'PAB_LAT', 'App_Lon_Sun',  'I_dRA_cosD', 'I_d_DEC__dt',
                    'Sky_motion', 'Sky_mot_PA', 'RelVel_ANG', 'Lun_Sky_Brt', 'sky_SNR',
                    'sat_primary_X', 'sat_primary_Y', 'a_app_Azi', 'a_app_Elev',
                    'ang_sep', 'T_O_M', 'MN_Illu_',
                    'd_DEC__slash_dt', 'd_ELV__slash_dt',
                    'Theta', 'Area_3sig',
                    'I_d_DEC__slash_dt',
                    ]
        str_col = ['_slash_r', 'Cnst', 'L_Ap_Sid_Time', 'L_Ap_SOL_Time', 'L_Ap_Hour_Ang',
                'moon_presence']

        ut_col = ['Date___UT___HR_MN']
        jd_col = 'Date_________JDUT'

        coords_col = {'R_A_______ICRF______DEC': 'ICRF',
                    'R_A____a_apparent___DEC': 'apparent',
                    'RA__ICRF_a_apparnt__DEC': 'ICRF_app',
                    }
        sexagesimal_col = ['L_Ap_Sid_Time', 'L_Ap_SOL_Time', 'L_Ap_Hour_Ang',
                        ]
        two_values_col = ['X__sat_primary__Y', 'Azi_____a_app____Elev']
        slash_col = ['ang_sep_slash_v', 'T_O_M_slash_MN_Illu_']

        convert_dict = ({k: float for k in float_col} |
                        {k: str for k in str_col + ut_col + list(coords_col.keys()) +
                        sexagesimal_col + two_values_col + slash_col})

        def month_name_to_number(string):
            for idx, month in enumerate(['Jan', "Feb", "Mar", "Apr", "May", "Jun",
                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
                if month in string:
                    return string.replace(month, f'{idx + 1:02d}')
            else:
                raise ValueError("No month name found")

        def change_names(string):
            string, _ = re.subn(r'1(\D)', r'one\1', string)
            string, _ = re.subn(r'399', 'earth', string)
            string, _ = re.subn(r'[%*().:-]', '_', string)
            string, _ = re.subn(r'/', '_slash_', string)
            return string

        previous = ""

        moon_presence = True
        while True:
            line = ephemeris.pop(0)
            if len(ephemeris) == 0:
                raise ValueError("No Ephemeris info: it should be surrounded by $$SOE and $$EOE")
            if re.match(r"\*+ *", line):
                continue
            if re.match(r"\$\$SOE", line):
                break
            if re.match(r'Center-site name', line):
                if 'GEOCENTRIC' in line:
                    moon_presence = False
            previous = line
        spaces = 0
        col_seps = re.split(r'( +)', previous.rstrip())
        if col_seps[0] == '':
            col_seps.pop(0)
        date = col_seps.pop(0)

        chars = len(date)
        col_names = [change_names(date)]
        cum_chars = chars
        col_ends = [cum_chars]

        if moon_presence:
            spaces = len(col_seps.pop(0)) - 3
            col_names.append('moon_presence')
            cum_chars += 3
            col_ends.append(cum_chars)

        for val in col_seps:
            if val[0] == '\0':
                break
            elif val[0] != ' ':
                chars = len(val) + spaces
                col_names.append(change_names(val))
                cum_chars += chars
                col_ends.append(cum_chars)
            else:
                spaces = len(val)

        incoming = []
        while True:
            line = ephemeris.pop(0)
            if len(ephemeris) == 0:
                raise ValueError("No Ephemeris info: it should be surrounded by $$SOE and $$EOE")

            if re.match(r"\$\$EOE", line.rstrip()):
                break

            incoming.append(line.replace("n.a.", "    ").rstrip())

        # dataframe with each field separated
        table = io.ascii.read(incoming,
                            format='fixed_width_no_header',
                            names=col_names,
                            col_ends=col_ends,
                            converters=convert_dict,
                            )

        def splitter(iterable, dtype=None, n_elements=None, separator=None):
            if dtype is None:
                def dtype(data):
                    return data

            ret = []
            mask = []
            for value in iterable:
                if ma.is_masked(value):
                    ret.append([0] * n_elements)
                    mask.append([True] * n_elements)
                    continue

                ret.append([dtype(v) for v in value.split(separator)])
                if n_elements is None:
                    n_elements = len(ret[-1])
                mask.append([False] * n_elements)

            if isinstance(iterable, MaskedColumn):
                return ma.masked_array(ret, mask=mask)
            else:
                return np.array(ret)

        # convert sexagesimal coordinates to float
        for coord, name in coords_col.items():
            if coord not in table.columns:
                continue

            coords = splitter(table[coord], dtype=float, n_elements=6)
            table[f'ra_{name}'] = coords[:, 0] + (coords[:, 1] + coords[:, 2] / 60) / 60
            sign = np.sign(coords[:, 3])
            sign += sign == 0  # dec == 0 is positive +1, not 0
            table[f'dec_{name}'] = sign * (np.abs(coords[:, 3]) + (coords[:, 4] + coords[:, 5] / 60) / 60)

        # convert two values into their own columns
        for column in two_values_col:
            if column not in table.columns:
                continue
            local_pattern = re.compile(r"([a-zA-Z]+?)_+(.+?)_+([a-zA-Z]+?)$")
            match = re.match(local_pattern, column)
            left, right = f"{match[2]}_{match[1]}", f"{match[2]}_{match[3]}",
            vals = splitter(table[column], n_elements=2)
            table[left] = vals[:, 0]
            table[right] = vals[:, 1]

        # convert slash-separated two values
        for column in slash_col:
            if column not in table.columns:
                continue
            local_pattern = re.compile(r"(.+)_slash_(.+)$")
            match = re.match(local_pattern, column)
            left, right = match[1], match[2]
            vals = splitter(table[column], n_elements=2, separator='/')
            table[left] = vals[:, 0]
            table[right] = vals[:, 1]

        # convert sexagesimal column to float
        for column in sexagesimal_col:
            if column not in table.columns:
                continue
            sexagesimal = splitter(table[column], n_elements=3, dtype=float)
            sign = np.sign(sexagesimal[:, 0])
            sign += sign == 0  # 0 is positive
            table[f'{column}_float'] = sign * (sexagesimal[:, 0] + (sexagesimal[:, 1] + sexagesimal[:, 2] / 60) / 60)

        # convert UT date to JD
        for column in ut_col:
            if column not in table.columns:
                continue

            table[jd_col] = [apt.Time(month_name_to_number(s.replace(" ", "T"))).jd for s in table[column]]

        table['jd'] = table[jd_col]

        return table

    @staticmethod
    def read_jpl(specification):
        """
        Combined method to get and parse JPL ephemeris data in one step.
        
        Parameters
        ----------
        specification : str or dict
            Ephemeris specification. See get_jpl_ephemeris for details.
            
        Returns
        -------
        astropy.table.Table
            Table with named columns containing the parsed ephemeris data
        """
        ephemeris = HorizonsInterface.get_jpl_ephemeris(specification)
        return HorizonsInterface.parse_jpl_ephemeris(ephemeris)

    @staticmethod
    def path_from_jpl(body,
                    observer,
                    times: apt.Time,
                    ):
        """
        Get values sorted by Julian date for the movement of a Solar System body.
        
        Parameters
        ----------
        body : str or int
            Name or ID number of the body to query
        observer : str
            Name of the observatory or observing location
        times : astropy.time.Time
            Time or array of times at which to calculate ephemeris
            
        Returns
        -------
        astropy.table.Table
            Table containing ephemeris data with an added 'skycoord' column
            containing SkyCoord objects for easy coordinate handling
        """
        time_spec = HorizonsInterface.jpl_times_from_time(times)
        site = apc.EarthLocation.of_site(observer)

        request = HorizonsInterface.jpl_body_from_str(body) | time_spec | HorizonsInterface.jpl_observer_from_location(site)
        ret = HorizonsInterface.read_jpl(request)

        ret['skycoord'] = apc.SkyCoord(ret['ra_ICRF'], ret['dec_ICRF'], unit=(u.hourangle, u.degree))

        return ret


    @staticmethod
    def jpl_body_from_str(body):
        """
        Convert a body name or ID to the format required by JPL Horizons.
        
        Parameters
        ----------
        body : str or int
            Name or ID of the solar system body
            
        Returns
        -------
        dict
            Dictionary with the 'COMMAND' key set to the body ID
            
        Raises
        ------
        ValueError
            If the body is not recognized
        """
        bodies = {'mercury': 199,
                'venus': 299,
                'moon': 301,
                'luna': 301,
                'mars': 499,
                'jupiter': 599,
                'saturn': 699,
                'uranus': 799,
                'neptune': 899,
                }
        match body:
            case int():
                pass
            case str():
                body = bodies[body.lower()]
            case a:
                raise ValueError(f"Invalid value in body ({a})")

        return {'COMMAND': body}

    @staticmethod
    def jpl_times_from_time(times: str | apt.Time):
        """
        Convert astropy Time objects to the format required by JPL Horizons.
        
        Parameters
        ----------
        times : str or astropy.time.Time
            Time or times for which to request ephemeris data
            
        Returns
        -------
        dict
            Dictionary with keys and values formatted for JPL Horizons TLIST
            
        Raises
        ------
        ValueError
            If more than 10,000 times are provided (Horizons limit)
        """
        if isinstance(times, str):
            times = apt.Time(times, format='isot', scale='utc')
        if times.isscalar:
            times = apt.Time([times])
        if len(times) > 10000:
            raise ValueError("Horizon's interface only accepts a maximum of 10,000 discrete times to provide to TLIST")

        times_str = " ".join([f"'{s}'" for s in times.jd])
        return {'TLIST_TYPE': 'JD',
                'TIME_TYPE': 'UT',
                'TLIST': times_str}
    
    @staticmethod
    def jpl_observer_from_location(site):
        """
        Convert an astropy EarthLocation to the format required by JPL Horizons.
        
        Parameters
        ----------
        site : astropy.coordinates.EarthLocation
            Location of the observer
            
        Returns
        -------
        dict
            Dictionary with keys and values formatted for JPL Horizons observer location
        """
        return {'CENTER': "'coord@399'",
                'COORD_TYPE': "'GEODETIC'",
                'SITE_COORD': (f"'{site.lon.degree:.2f}, {site.lat.degree:.2f}," +
                            f" {site.height.to(u.km).value:.2f}'"),
                }


def body_path(body,
              observer,
              times: apt.Time,
              use_jpl: bool = False,
              ):
    """

    Parameters
    ----------
    use_jpl
    body
    observer
    times
    """

    if use_jpl:
        return HorizonsInterface.path_from_jpl(body, observer, times)

    site = apc.EarthLocation.of_site(observer)
    body_object = apc.get_body(body, times, location=site)

    return QTable([times, times.jd, body_object,
                   body_object.ra.degree, body_object.dec.degree],
                  names=['time', 'jd', 'skycoord', 'ra', 'dec'])

