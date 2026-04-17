from dataclasses import dataclass
import numpy as np
from scipy.constants import G as G_SI, parsec as PC_SI

# Define fixed unit conversions (SI -> Galactic)
KPC_TO_M = 1e3 * PC_SI
M_SUN_TO_KG = 1.988e30


@dataclass(frozen=True)
class Cosmology:
    """
    Immutable container for cosmological parameters.
    Automatically derives density parameters in Galactic units (M_sun / kpc^3).
    """
    # Fundamental parameters (Tunable)
    h: float = 0.7              # Hubble Parameter (H0 = 100h)
    omega_m: float = 0.3        # Matter density parameter
    delta_vir: float = 200.0    # Virial overdensity definition

    # Derived properties
    @property
    def rho_crit(self) -> float:
        """
        Critical density of the universe in M_sun / kpc^3.
        rho_c = 3 H_0^2 / (8 pi G)
        """
        # H0 in SI units (s^-1); H0 = 100 h km/s/Mpc
        H0_si = (100 * self.h * 1000) / (1e6 * PC_SI)

        # rho_c in SI (kg / m^3)
        rho_c_si = (3 * H0_si**2) / (8 * np.pi * G_SI)

        # Convert from kg / m^3 to M_sun / kpc^3
        rho_c_galactic = rho_c_si / M_SUN_TO_KG * (KPC_TO_M**3)
        return rho_c_galactic


# %% Example usage
if __name__ == "__main__":
    # Instantiate the default cosmology and print derived parameters
    cosmo = Cosmology()
    print(f"Cosmological parameters:"
          f"h={cosmo.h}, Omega_m={cosmo.omega_m}, Delta_vir={cosmo.delta_vir}")
    print(f"Derived Critical Density (M_sun/kpc^3): {cosmo.rho_crit:.2e}")

    # Can also create a custom cosmology, e.g. for Planck 2018 parameters
    planck_cosmo = Cosmology(h=0.674, omega_m=0.313)
    print("Planck Cosmology Critical Density (M_sun/kpc^3):"
          f"{planck_cosmo.rho_crit:.2e}")
