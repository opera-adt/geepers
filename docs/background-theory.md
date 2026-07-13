# Background theory

The methods implemented in geepers, with just enough math to interpret
the outputs. References at the bottom.

## Projecting GPS onto the radar line of sight

InSAR measures the projection of the 3-D displacement onto the
satellite line of sight. With the LOS unit vector
$\mathbf{u} = (u_E, u_N, u_U)$ pointing from ground to satellite,

$$
d_\text{LOS} = u_E\, d_E + u_N\, d_N + u_U\, d_U .
$$

The GPS LOS uncertainty follows from the full ENU covariance
$\Sigma$ (variances and correlations from the GPS solution):

$$
\sigma_\text{LOS}^2 = \mathbf{u}^\mathsf{T} \Sigma\, \mathbf{u},
$$

implemented in `geepers.uncertainty.get_sigma_los`. When the InSAR
stack stores phase in radians, displacement is converted using the
sensor wavelength passed to the workflow (Sentinel-1 C-band by
default; pass `--wavelength 0.2384` for NISAR L-band).

## MIDAS: robust velocities (`geepers.midas`)

MIDAS (Blewitt et al., 2016) is a Theil-Sen-type estimator customized
for GNSS. It forms data pairs separated by (almost exactly) one year —
so seasonal cycles cancel by construction — optionally never spanning a
known step epoch, and takes the median $\tilde v$ of the pair
velocities. Outliers beyond $2\sigma$ (from the scaled MAD,
$\sigma = 1.4826\,\mathrm{MAD}$) are removed and the median recomputed.
The velocity uncertainty is

$$
\sigma_v = 3 \sqrt{\frac{\pi}{2}} \; \frac{\sigma}{\sqrt{N/4}},
$$

where $N$ is the number of retained pairs; the factor $\sqrt{\pi/2}$
converts a median-based estimate to a standard-error equivalent, the
$/4$ accounts for pair correlation, and the empirical factor 3 absorbs
temporally correlated noise. MIDAS is insensitive to steps, outliers,
and seasonality, but its uncertainty is a calibrated approximation
rather than a noise-model-based estimate.

## Trend estimation with colored noise (`geepers.trend`)

GNSS residuals are not white: their power spectra behave as
$P(f) \propto f^{\kappa}$ with spectral index $\kappa$ typically
between $-1$ (flicker) and $0$ (white). Ignoring this makes
least-squares velocity uncertainties 5-10× too small (Williams, 2003).

geepers fits the deterministic model (polynomial, periodic terms,
offsets, postseismic decays) *jointly* with a two-component noise model

$$
\mathbf{C}(\kappa, \phi) = \sigma^2 \left[ \phi\, \mathbf{I} +
(1-\phi)\, \mathbf{C}_\text{PL}(\kappa) \right],
$$

where $\phi$ is the white-noise fraction. The power-law covariance uses
the fractional-differencing formulation (Hosking, 1981): with
$d = -\kappa/2$ and coefficients
$\psi_0 = 1,\; \psi_i = \psi_{i-1}\,(d + i - 1)/i$,

$$
\mathbf{C}_\text{PL} = \mathbf{U}\mathbf{U}^\mathsf{T},
\qquad U_{ij} = \psi_{i-j} \; (i \ge j),
$$

valid also for non-stationary noise ($\kappa \le -1$), evaluated at the
observed epochs only, which handles data gaps exactly (Bos et al.,
2013). For fixed $(\kappa, \phi)$ the model parameters come from
generalized least squares and $\sigma^2$ is profiled out analytically;
the two noise parameters are found by numerically maximizing the
(restricted) log-likelihood

$$
-2\ln L = \ln\det\mathbf{C} + (n - p)\ln\hat\sigma^2 +
\ln\det(\mathbf{A}^\mathsf{T}\mathbf{C}^{-1}\mathbf{A}) + \text{const},
$$

where the last term is the RMLE correction for the $p$ estimated model
parameters. The velocity uncertainty is then read from
$\hat\sigma^2 (\mathbf{A}^\mathsf{T}\mathbf{C}^{-1}\mathbf{A})^{-1}$ —
it inherits the temporal correlation of the noise.

The implementation is clean-room from the published equations and was
validated against HectorP (Bos et al.): velocities agree to
~0.005 mm/yr in well-conditioned cases; in the strongly
flicker-dominated regime sub-1σ differences remain, traceable to
HectorP's GGM regularization of the pure power law.

### Fast spectral estimation (Whittle likelihood)

For long series the O(n³) exact likelihood is replaced by the Whittle
approximation: the periodogram of the (detrended) residuals is fit to
the analytic power-law + white spectrum

$$
P(f) = \sigma_{pl}^2 \left| 2 \sin(\pi f \Delta t) \right|^{\kappa}
 + \sigma_w^2,
$$

whose likelihood is a sum over frequencies — O(n log n). Noise
parameters from the spectrum then feed one GLS solve for the trend and
its uncertainty. Slightly coarser $\kappa$ estimates than the exact
method, but usable on decades of daily data and whole networks
(`estimate_trend_many`).

## Common-mode error (`geepers.cme`)

Stack the detrended residuals of a network into a matrix (epochs ×
stations); its leading principal component(s) capture the coherent
regional signal — reference-frame wobble and large-scale
atmosphere/loading (Wdowinski et al., 1997 "stacking" generalized to
PCA). Removing the reconstructed common mode
$\mathbf{u}_k \mathbf{s}_k^\mathsf{T}$ typically shrinks per-station
scatter by 30-50% without touching station-specific signals.

## Step detection (`geepers.steps`)

For each candidate epoch, two models are fit in a sliding window
centered on it: a line, and a line plus a step. The epoch is flagged
when the AIC improvement of the step model exceeds a threshold —
a parameter-counted version of the classic two-sample test that is
robust to the trend itself. Detections closer than a minimum separation
are merged, keeping the strongest.

## Strain rates (`geepers.strain`)

From a gridded horizontal velocity field, the 2D infinitesimal
strain-rate tensor is

$$
\dot\varepsilon_{xx} = \partial_x v_e,\quad
\dot\varepsilon_{yy} = \partial_y v_n,\quad
\dot\varepsilon_{xy} = \tfrac12(\partial_y v_e + \partial_x v_n),
$$

with gradients converted from per-degree to per-meter using spherical
metric factors ($R\cos\varphi$ for longitude). Derived scalars:
dilatation $\dot\varepsilon_{xx}+\dot\varepsilon_{yy}$, maximum shear
$\sqrt{(\dot\varepsilon_{xx}-\dot\varepsilon_{yy})^2/4+\dot\varepsilon_{xy}^2}$,
rotation $\tfrac12(\partial_x v_n - \partial_y v_e)$, and the second
invariant.

## Velocity variability metrics (`geepers.variability`)

**Temporal**: MIDAS velocities are computed in half-overlapping sliding
windows of every length from 3 years up to 75% of the record; the
metric is the robust spread around the full-series velocity,
$\sqrt{\mathrm{median}\left((v_\text{win} - v_\text{full})^2\right)}$,
per component. Stations whose velocity depends on the observation
window get large values — a warning sign for transients or unmodeled
steps.

**Spatial**: for each station, the RMS and MAD of the velocity
differences with its Delaunay-network neighbors; gross outliers stand
out immediately.

**Spatial structure function (SSF)**: pairwise velocity differences
binned by separation; the inverted, normalized curve (1 at zero
distance, 0 at 180°) measures how coherence decays with distance
(Hammond et al., 2016, eqs. 1-2).

## GPS Imaging (`geepers.gps_imaging`)

A robust interpolator built from three parts (Hammond et al., 2016):

1. the **SSF** as above, with the zero-lag bin anchored at the median
   measurement uncertainty and the per-bin scatter forced non-decreasing
   with distance;
2. **Delaunay neighborhoods** — each point is estimated from the
   stations it is connected to in the triangulation (optionally
   expanded to all stations within the median neighbor distance;
   Kreemer et al., 2020);
3. the **weighted median** of the neighborhood velocities, with weights
   $w_i = \mathrm{SSF}(d_i)/\sigma_i$.

Because the estimator is a median, single bad stations cannot leak into
the map; and because there is no explicit smoothing, sharp velocity
boundaries survive wherever stations constrain them. The geepers port
reproduces the reference MATLAB kernels exactly (bit-identical SSF and
weighted-median outputs on randomized tests).

## Least-squares collocation (`geepers.collocation`)

The geostatistical counterpart (Moritz, 1980). Observations are modeled
as signal plus noise, $\mathbf{z} = \mathbf{s} + \mathbf{n}$, with
signal covariance $\mathbf{C}_{ss}$ from a fitted isotropic model
(first-order Gauss-Markov by default, $C(d) = C_0 e^{-d/d_0}$) and
diagonal noise covariance $\mathbf{C}_{nn}$ from the station sigmas.
The estimates at the stations and at new points $p$ are

$$
\hat{\mathbf{s}} = \mathbf{C}_{ss}(\mathbf{C}_{ss}+\mathbf{C}_{nn})^{-1}\mathbf{z},
\qquad
\hat{\mathbf{s}}_p = \mathbf{C}_{ps}(\mathbf{C}_{ss}+\mathbf{C}_{nn})^{-1}\mathbf{z},
$$

with error covariance
$\mathbf{C}_{pp} - \mathbf{C}_{ps}(\mathbf{C}_{ss}+\mathbf{C}_{nn})^{-1}\mathbf{C}_{ps}^\mathsf{T}$
— uncertainties grow away from data automatically. For horizontal
velocity fields the east/north blocks are coupled by the angular
covariance factors of a rotational field on the sphere, so the
interpolation respects plate-rotation geometry. The model parameters
$(C_0, d_0)$ come from the binned empirical covariogram of the data,
with $C_0$ anchored at the noise-corrected variance
$\overline{z^2} - \overline{\sigma^2}$.

## References

- Blewitt, G., Kreemer, C., Hammond, W. C., & Gazeaux, J. (2016). MIDAS
  robust trend estimator for accurate GPS station velocities without
  step detection. *J. Geophys. Res. Solid Earth*, 121, 2054-2068.
  doi:10.1002/2015JB012552
- Bos, M. S., Fernandes, R. M. S., Williams, S. D. P., & Bastos, L.
  (2013). Fast error analysis of continuous GNSS observations with
  missing data. *J. Geod.*, 87(4), 351-360.
  doi:10.1007/s00190-012-0605-0
- Hammond, W. C., Blewitt, G., & Kreemer, C. (2016). GPS Imaging of
  vertical land motion in California and Nevada. *J. Geophys. Res.
  Solid Earth*, 121, 7681-7703. doi:10.1002/2016JB013458
- Hosking, J. R. M. (1981). Fractional differencing. *Biometrika*,
  68(1), 165-176. doi:10.1093/biomet/68.1.165
- Kreemer, C., Hammond, W. C., & Blewitt, G. (2020). A robust
  estimation of the 3-D intraplate deformation of the North American
  plate from GPS. *Geophys. Res. Lett.*, 47. doi:10.1029/2020GL087976
- Moritz, H. (1980). *Advanced Physical Geodesy*. Wichmann, Karlsruhe.
- Wdowinski, S., Bock, Y., Zhang, J., Fang, P., & Genrich, J.
  (1997). Southern California permanent GPS geodetic array:
  Spatial filtering of daily positions. *J. Geophys. Res.*, 102(B8),
  18057-18070. doi:10.1029/97JB01378
- Williams, S. D. P. (2003). The effect of coloured noise on the
  uncertainties of rates estimated from geodetic time series.
  *J. Geod.*, 76(9-10), 483-494. doi:10.1007/s00190-002-0283-4
