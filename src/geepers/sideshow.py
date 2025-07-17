"""JPL SIDESHOW GNSS data handling and downloading functionality."""


__all__ = [
    "get_stations_within_image",
    "load_station_enu",
    "read_station_llas",
]


SITE_LIST_URL = ("https://sideshow.jpl.nasa.gov/post/tables/table2.html",)
STATION_URL_BASE = (
    "https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/"
)
# e.g.
# https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/repro2018a/post/point/AB01.series
GPS_BASE_URL = f"{STATION_URL_BASE}{{station}}.series"
# https://sideshow.jpl.nasa.gov/post/tables/GNSS_Time_Series.pdf
# Time Series and Residual Format
# Column 1: Decimal_YR
# Columns 2-4: East(m) North(m) Vert(m)
# Columns 5-7: E_sig(m) N_sig(m) V_sig(m)
# Columns 8-10: E_N_cor E_V_cor N_V_cor
# Column 11: Time in Seconds past J2000
# Columns 12-17: Time in YEAR MM DD HR MN SS

STEPS_URL = "https://sideshow.jpl.nasa.gov/post/tables/table3.html"
# Break estimates and errors in mm.
# Break times in years.

#                            N             E             V      SN      SE      SV
# AB01 2013.6646        -9.335        -3.531         1.499   1.403   1.047   4.071
# AB01 2014.1793        -5.689         1.527         0.741   1.359   1.014   3.954


# FROM MINTPY
# def read_SIDESHOW_site_list(site_list_file: str):
#     """Return names and lon/lat values for JPL SIDESHOW GNSS stations."""
#     fc = np.loadtxt(site_list_file, comments="<", skiprows=9, dtype=str)
#     sites = {
#         "site": fc[::2, 0],
#         "lat": fc[::2, 2].astype(np.float32),
#         "lon": fc[::2, 3].astype(np.float32),
#     }
#     return sites
