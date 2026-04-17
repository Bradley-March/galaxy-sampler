from dataclasses import dataclass, fields
import numpy as np


class LogAccessor:
    """
    A lightweight proxy that intercepts attribute access, 
    fetches the value from the parent, and returns log10 of it.

    Usage: params.log.mass  -> returns np.log10(params.mass)
    """

    def __init__(self, parent):
        self._parent = parent

    def __getattr__(self, name):
        # Get the value from the parent (e.g., mass array)
        try:
            val = getattr(self._parent, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self._parent).__name__}' object has no attribute '{name}'")

        # Check validity
        if not isinstance(val, (np.ndarray, float, int)):
            raise TypeError(f"Cannot take log of non-numeric field '{name}'")

        # Return log10 (with safety)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log10(val)


@dataclass
class GalacticParams:
    """
    Base class to provides generic array manipulation methods & .log namespace.
    """

    @property
    def size(self):
        """Returns the length of the first field in the dataclass."""
        first_field = fields(self)[0].name
        return len(getattr(self, first_field))

    @property
    def log(self):
        """Provides access to log10 of parameters via .log.param_name"""
        return LogAccessor(self)

    def filter(self, mask):
        """
        Generic filter that returns a new instance of the same class,
        slicing all fields by the provided boolean mask.
        """
        # Inspect the child class to find what fields it has
        child_fields = fields(self)
        # Create a dictionary of {field_name: sliced_array}
        filtered_data = {
            f.name: getattr(self, f.name)[mask]
            for f in child_fields
        }
        # Return new instance of the child class (self.__class__)
        return self.__class__(**filtered_data)


@dataclass
class StellarDiscParams(GalacticParams):
    """
    Container for stellar disc parameters. Each attribute is an array of values.

    Attributes:
        mass                    : np.ndarray  # [M_sun]
        half_light_radius       : np.ndarray  # [kpc]
        oblateness              : np.ndarray  # [1]
        scale_radius            : np.ndarray  # [kpc]
        scale_height            : np.ndarray  # [kpc]
        density_normalisation   : np.ndarray  # [M_sun / kpc^2]
    """
    # Sampled parameters
    mass:                   np.ndarray
    half_light_radius:      np.ndarray
    oblateness:             np.ndarray

    # Density profile parameters
    scale_radius:           np.ndarray
    scale_height:           np.ndarray
    density_normalisation:  np.ndarray


@dataclass
class NFWHaloParams(GalacticParams):
    """
    Container for NFW halo parameters. Each attribute is an array of values.

    Attributes:
        mass                    : np.ndarray  # [M_sun]
        concentration           : np.ndarray  # [1]
        scale_radius            : np.ndarray  # [kpc]
        virial_radius           : np.ndarray  # [kpc]
        density_normalisation   : np.ndarray  # [M_sun / kpc^3]

    Note: mass is the virial mass (M_200c), not the total halo mass 
          (which is divergent for an NFW profile).
    """
    # Sampled parameters
    mass:                   np.ndarray
    concentration:          np.ndarray

    # Derived parameters
    virial_radius:          np.ndarray

    # Density profile parameters
    scale_radius:           np.ndarray
    density_normalisation:  np.ndarray


@dataclass
class GalaxyPopulation(GalacticParams):
    """
    Unified container for a full galaxy population.

    Attributes
    ----------
    disc : GalacticParams
        The baryonic component (i.e.., StellarDiscParams).
    halo : GalacticParams
        The dark matter component (i.e., NFWHaloParams).
    """
    disc: GalacticParams
    halo: GalacticParams

    @property
    def size(self):
        """
        Returns the size of the population. 
        Delegates to the disc component as the source of truth.
        """
        return self.disc.size

    def filter(self, mask):
        """
        Recursive filter.
        Applies the mask to the child class and returns a new GalaxyPopulation.
        """
        filtered_disc = self.disc.filter(mask)
        filtered_halo = self.halo.filter(mask)

        return GalaxyPopulation(disc=filtered_disc, halo=filtered_halo)


# %% Example Usage
if __name__ == "__main__":
    # Create a sample StellarDiscParams instance
    # (Note these are not physically meaningful values, just for demonstration)
    sdp = StellarDiscParams(
        mass=np.array([1e8, 1e9, 1e10, 1e11]),
        half_light_radius=np.array([1.5, 2.0, 2.5, 4.5]),
        oblateness=np.array([4.5, 5.5, 6.0, 7.5]),
        scale_radius=np.array([0.5, 1.0, 1.5, 2.5]),
        scale_height=np.array([0.1, 0.2, 0.3, 0.4]),
        density_normalisation=np.array([3e7, 1e8, 6e8, 2e9])
    )

    # Access log parameters
    print("Log Mass:", sdp.log.mass)

    # Filter example (keep only smaller mass galaxies)
    filtered_sdp = sdp.filter(sdp.mass <= 1e10)
    print("Filtered Mass:", filtered_sdp.mass)

    # Create a sample NFWHaloParams instance
    # (Note these are not physically meaningful values, just for demonstration)
    dmp = NFWHaloParams(
        mass=np.array([4e10, 1e11, 3e11, 1e13]),
        concentration=np.array([11.5, 10.5, 9.4, 6.6]),
        virial_radius=np.array([70, 95, 140, 445]),
        scale_radius=np.array([6, 9, 15, 67]),
        density_normalisation=np.array([8e6, 7e6, 5e6, 2e6])
    )

    # Create a GalaxyPopulation instance
    population = GalaxyPopulation(disc=sdp, halo=dmp)
    print("Population Size:", population.size)
    print("Population Disc Masses:", population.disc.mass)
    print("Population Halo Masses:", population.halo.mass)
    # Filter the population (e.g., keep only galaxies with disc mass <= 1e10)
    filtered_population = population.filter(population.disc.mass <= 1e10)
    print("Filtered Population Size:", filtered_population.size)
    print("Filtered Population Disc Masses:", filtered_population.disc.mass)
    print("Filtered Population Halo Masses:", filtered_population.halo.mass)
