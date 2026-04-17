import numpy as np
from structures import NFWHaloParams, StellarDiscParams, GalaxyPopulation
from relations import MassRelations, NFWHaloRelations, StellarRelations
from cosmology import Cosmology
from numerics import sample_from_pdf


class GalaxyPipeline:
    """
    A modular pipeline to generate a galactic population from empirical relations.

    The pipeline is designed to be flexible, allowing users to inject custom
    PDFs for sampling, and to toggle intrinsic scatter in the relations.
    It also supports a custom cosmology for halo calculations.

    Workflow:
    1. Sample stellar masses from an SMF (e.g., Schechter function).
    2. Invert the SHMR to get corresponding virial masses.
    3. Use the sampled masses to generate disc and halo density parameters via
       empirical relations, applying scatter if desired.
    4. Package everything into a unified GalaxyPopulation object.
    """

    def __init__(self, cosmo: Cosmology = None):
        """
        Initialise the pipeline with a specific cosmology.
        If None, loads the default (h=0.7, Omega_m=0.3).
        """
        self.cosmo = cosmo if cosmo is not None else Cosmology()

    def sample_from_smf(self, n_samples,
                        mass_min=1e8, mass_max=1e12,
                        smf_pdf=MassRelations.pdf_schechter_log, **smf_kwargs):
        """
        Sample a population of stellar masses from an SMF, given by `smf_pdf`.

        Parameters
        ----------
        n_samples           : int
        mass_min, mass_max  : float     # [M_sun, M_sun]
            Bounds to sample between.
        smf_pdf             : function
            SMF (pdf) function e.g. MassRelations.pdf_schechter_log,
            MassRelations.pdf_double_schechter_log,
            MassRelations.pdf_bounded_log_normal,
            or a custom user-defined function.
        smf_kwargs          : dict
            Additional parameters for the chosen SMF (e.g., alpha, M_disc, phi_star).

        Returns
        -------
        np.ndarray of sampled stellar masses [M_sun]
        """
        # convert mass bounds to log-space for sampling
        bounds = (np.log10(mass_min), np.log10(mass_max))

        # Sample log-masses using inverse transform sampling
        sampled_log_masses = sample_from_pdf(
            n_samples, smf_pdf, bounds, **smf_kwargs)

        # Convert back to linear space
        stellar_masses = 10**sampled_log_masses
        return stellar_masses

    def generate_stellar_disc_population(self, seed_stellar_mass, scatter=True):
        """
        Generates a full population of stellar disc parameters from a
        seed of stellar mass(es).

        Parameters
        ----------
        seed_stellar_mass   : array_like   # [M_sun]
        scatter             : bool
            If True, applies intrinsic scatter to sampled parameters.

        Returns
        -------
        StellarDiscParams object
        """
        M_disc = np.asarray(seed_stellar_mass)

        # Mass -> Half-Light Radius
        R_hl = StellarRelations.get_half_light_radius(M_disc, scatter=scatter)

        # Half-Light -> Scale Radius
        R_disc = StellarRelations.get_scale_radius(R_hl)

        # Scale Radius -> Oblateness (q)
        q = StellarRelations.get_oblateness(R_disc, scatter=scatter)

        # Derive remaining density profile parameters
        z_disc = StellarRelations.get_scale_height(R_disc, q)
        Sigma_disc = StellarRelations.get_density_normalisation(M_disc, R_disc)

        # Package into container
        return StellarDiscParams(
            mass=M_disc,
            half_light_radius=R_hl,
            oblateness=q,
            scale_radius=R_disc,
            scale_height=z_disc,
            density_normalisation=Sigma_disc
        )

    def generate_nfw_halo_population(self, seed_virial_mass, scatter=True):
        """
        Generates a full population of NFW Halos from a seed of Virial Masses.

        Parameters
        ----------
        seed_virial_mass      : array_like    # [M_sun]
        scatter             : bool
            If True, applies intrinsic scatter to sampled parameters.

        Returns
        -------
        NFWHaloParams object
        """
        M_vir = np.asarray(seed_virial_mass)

        # Mass -> Concentration
        c_vir = NFWHaloRelations.get_concentration(
            M_vir, h=self.cosmo.h, scatter=scatter)

        # Mass + Concentration -> Scale Radius & Virial Radius
        r_vir = NFWHaloRelations.get_virial_radius(
            M_vir, self.cosmo.rho_crit, self.cosmo.delta_vir)
        r_nfw = NFWHaloRelations.get_scale_radius(r_vir, c_vir)

        # Mass + Scale Radius -> Density Normalisation
        rho_nfw = NFWHaloRelations.get_density_normalisation(
            M_vir, r_nfw, c_vir)

        # Package into container
        return NFWHaloParams(
            mass=M_vir,
            concentration=c_vir,
            scale_radius=r_nfw,
            virial_radius=r_vir,
            density_normalisation=rho_nfw
        )

    def generate_population(
        self,
        n_samples,
        scatter=True,
        stellar_mass_min=1e8, stellar_mass_max=1e12,
        smf_pdf=MassRelations.pdf_schechter_log,
        **smf_kwargs
    ):
        """
        Generate a statistically consistent population of galaxies, with
        parameters for disc and halos profiles.

        Parameters
        ----------
        n_samples       : int
        scatter         : bool
            If True, applies intrinsic scatter to all relations.
        stellar_mass_min, stellar_mass_max : float
            Bounds for sampling stellar masses from the SMF.
        smf_pdf         : callable
            The PDF function to sample stellar masses from.
            Default: MassRelations.pdf_schechter_log
        **smf_kwargs    : dict
            Keyword arguments passed to smf_pdf (e.g., alpha, log_M_star).

        Returns
        -------
        GalaxyPopulation
        """
        # Sample seed stellar masses from the SMF
        M_disc = self.sample_from_smf(
            n_samples,
            mass_min=stellar_mass_min, mass_max=stellar_mass_max,
            smf_pdf=smf_pdf, **smf_kwargs)

        # Use numerically inverted SHMR to get seed halo masses
        M_vir = MassRelations.get_virial_mass_from_stellar_mass(
            M_disc, scatter=scatter)

        # Generate component populations from seed masses
        disc_params = self.generate_stellar_disc_population(
            M_disc, scatter=scatter)
        halo_params = self.generate_nfw_halo_population(
            M_vir, scatter=scatter)

        # Return unified population
        return GalaxyPopulation(disc=disc_params, halo=halo_params)


# %% Example usage
if __name__ == "__main__":
    # Set rng seed for reproducibility
    np.random.seed(42)

    # Initialise pipeline with default cosmology
    pipeline = GalaxyPipeline()

    # Generate a population of 1000 galaxies with default SMF parameters
    population = pipeline.generate_population(n_samples=1000, scatter=True)

    # First 5 stellar/halo masses for the generated population
    print(f"Generated a population of {population.size} galaxies.")
    print(f"Sample disc masses (M_sun): {population.disc.mass[:5]}")
    print(f"Sample halo masses (M_sun): {population.halo.mass[:5]}")

    # Can also use each branch independently, starting from some seed mass array
    seed_stellar_masses = np.array([1e9, 1e10, 1e11])
    disc_population = pipeline.generate_stellar_disc_population(
        seed_stellar_masses)
    # likewise for dark matter halos
    seed_virial_masses = np.array([1e11, 1e12, 1e13])
    halo_population = pipeline.generate_nfw_halo_population(seed_virial_masses)

    # We can also use the pipeline to fetch a range of maximally typical
    # galaxies, by ignoring the scatter in empirical relations.
    typical_disc_population = pipeline.generate_stellar_disc_population(
        seed_stellar_masses[:5], scatter=False)
    print(typical_disc_population)

    # By default, the pipeline uses a Schechter function SMF.
    # A custom PDF can be injected, e.g. a log-normal:
    population = pipeline.generate_population(
        n_samples=1000,
        smf_pdf=MassRelations.pdf_bounded_log_normal,
        mu=10.6, sigma=0.35, lower_cutoff=8.0, upper_cutoff=11.0)
    # and the domain of the SMF can be adjusted via stellar_mass_min/max:
    population = pipeline.generate_population(
        n_samples=1000, stellar_mass_min=1e9, stellar_mass_max=1e11)
