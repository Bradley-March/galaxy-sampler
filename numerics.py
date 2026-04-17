import numpy as np
from scipy.interpolate import interp1d


def sample_from_pdf(n_samples, pdf_func, bounds, resolution=10_000, **pdf_kwargs):
    """
    Generic Inverse Transform Sampler. Used to sample from general PDFs.

    Parameters
    ----------
    n_samples : int
        Desired number of samples to draw.
    pdf_func : callable
        Function signature: pdf_func(x, **kwargs) -> density_array
    bounds : tuple
        (min_x, max_x) integration range.
    resolution : int
        Grid size for CDF integration.
    **pdf_kwargs : dict
        Arguments passed directly to pdf_func.

    Returns
    -------
    sampled_x : np.ndarray of shape (n_samples,)
        Samples drawn from the specified PDF.
    """
    # Create grid & evaluate PDF
    x_grid = np.linspace(*bounds, resolution)
    pdf_values = pdf_func(x_grid, **pdf_kwargs)

    # Integrate to CDF & normalise to [0, 1]
    cdf_values = np.cumsum(pdf_values)
    cdf_values = cdf_values / cdf_values[-1]

    # Sample uniform variable to map onto CDF
    u = np.random.rand(n_samples)

    # Interpolate x from u. fill_value ensures u=[0, cdf_values[0]) maps to x_min
    inverse_cdf = interp1d(cdf_values, x_grid, kind='linear',
                           bounds_error=False, fill_value=bounds)
    sampled_x = inverse_cdf(u)

    return sampled_x


class NumericalInverter:
    """
    Helper to invert monotonic relations using interpolation.
    Supports optional log-space transformations for numerical stability.
    """

    def __init__(self, forward_func, bounds, resolution=10_000, input_log=False, output_log=False):
        """
        Parameters
        ----------
        forward_func : callable
            The function y = f(x) to invert.
        bounds : tuple
            (min_x, max_x) range for the lookup grid.
        resolution : int
            Number of grid points.
        input_log : bool
            If True, the x-grid is created in log space, and interpolation is done on log(x).
        output_log : bool
            If True, interpolation is done on log(y).
        """
        self.input_log = input_log
        self.output_log = output_log

        # Create x grid in log/linear space
        if self.input_log:
            # Grid is uniform in log space (e.g., 1e8, 1e9, 1e10...)
            self.x_grid_vals = np.logspace(*np.log10(bounds), resolution)
            interp_x = np.log10(self.x_grid_vals)
        else:
            # Grid is uniform in linear space (e.g., 8, 9, 10...)
            self.x_grid_vals = np.linspace(*bounds, resolution)
            interp_x = self.x_grid_vals

        # Compute forward y
        self.y_grid_vals = forward_func(self.x_grid_vals)

        # Prepare y for interpolation
        if self.output_log:
            interp_y = np.log10(self.y_grid_vals)
        else:
            interp_y = self.y_grid_vals

        # Create inverse interpolator (y -> x)
        self.interp_func = interp1d(
            interp_y, interp_x, kind='cubic', fill_value="extrapolate")

    def __call__(self, y):
        # Transform input if required
        y_in = np.log10(y) if self.output_log else y

        # Interpolate using inverse interpolator
        x_out = self.interp_func(y_in)

        # Transform output if required
        return 10**x_out if self.input_log else x_out
