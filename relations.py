import numpy as np
from numerics import NumericalInverter


class MassRelations:
    """
    Stateless physics engine for galactic Mass-to-Mass relations.
    Includes SHMR and SMFs (single/double Schechter, and bounded log normal).

    All functions accept array-like inputs and return numpy arrays.
    """

    @staticmethod
    def get_stellar_mass_from_virial_mass(
        virial_mass,
        scatter=False,
        sigma_log_M_disc=0.15,
        N=0.0351, M1=10**11.59, beta=1.376, gamma=0.608
    ):
        """
        Calculate stellar mass from halo mass (SHMR).

        Relation: Moster et al. (2013) double power-law.
            M_disc = 2 * N * M_vir * [ (M_vir/M1)^-beta + (M_vir/M1)^gamma ]^-1

        Parameters
        ----------
        virial_mass           : array_like    # [M_sun]
        scatter : bool
            If True, applies log-normal scatter to the output stellar mass.
        sigma_log_M_disc         : float         # [dex]
            Scatter in log10(M_disc) at fixed halo mass.
        N, M1, beta, gamma  : float         # [1, M_sun, 1, 1]
            Fitting parameters (default: Table 1 of Moster et al. 2013, z=0).

        Returns
        -------
        stellar_mass        : np.ndarray    # [M_sun]
        """
        M_vir = np.asarray(virial_mass)

        # Term inside brackets: (M/M1)^-beta + (M/M1)^gamma
        x = M_vir / M1
        denominator = x**(-beta) + x**(gamma)

        # Calculate mean relation
        M_disc = 2 * N * M_vir / denominator

        # Apply scatter
        if scatter:
            # Scatter is in log10(Stellar Mass)
            noise = np.random.normal(
                loc=0.0, scale=sigma_log_M_disc, size=M_vir.shape)
            # M_final = 10^(log(M_mean) + noise) = M_mean * 10^noise
            return M_disc * 10**noise

        return M_disc

    @staticmethod
    def get_log_shmr_slope(virial_mass, M1, beta, gamma):
        """
        Calculates d(log M_disc) / d(log M_vir) analytically. 
        Assumes the Moster et al. 2013 double power-law form for the SHMR.

        M_disc = 2 * N * M_vir * [ x^-beta + x^gamma ]^-1
        where x = M_vir / M1
        => d log(M_disc) / d log(M_vir) 
            = 1 - (gamma * x^gamma - beta * x^-beta) / (x^gamma + x^-beta)


        Parameters
        ----------
        virial_mass   : array_like    # [M_sun]
        M1, beta, gamma : float     # [M_sun, 1, 1]
            Fitting parameters (default: Table 1 of Moster et al. 2013, z=0).

        Returns
        -------
        slope       : np.ndarray    # [1]
            Local slope of the log-SHMR at the given halo mass.
        """
        M_vir = np.asarray(virial_mass)
        x = M_vir / M1

        term_beta = x**(-beta)
        term_gamma = x**(gamma)

        numerator = gamma * term_gamma - beta * term_beta
        denominator = term_gamma + term_beta

        return 1 - numerator / denominator

    @staticmethod
    def get_virial_mass_from_stellar_mass(
        stellar_mass,
        scatter=False,
        sigma_log_M_disc=0.15,  # (Forward scatter)
        N=0.0351, M1=10**11.59, beta=1.376, gamma=0.608
    ):
        """
        Calculate halo mass from stellar mass (numerically inverted SHMR).
        Assumes the Moster et al. 2013 double power-law form for the SHMR.

        Parameters
        ----------
        stellar_mass        : array_like    # [M_sun]
        scatter             : bool
            If True, applies reverse scatter to the retrieved halo mass.
        sigma_log_M_disc    : float         # [dex]
            Scatter in log10(M_disc) at fixed halo mass (forward scatter).
        N, M1, beta, gamma  : float         # [1, M_sun, 1, 1]
            Fitting parameters (default: Table 1 of Moster et al. 2013, z=0).

        Returns
        -------
        virial_mass           : np.ndarray    # [M_sun]
        """
        M_disc = np.asarray(stellar_mass)

        #  Define forward model wrapper (takes M_vir and return M_disc)

        def forward(M_vir):
            return MassRelations.get_stellar_mass_from_virial_mass(
                M_vir, scatter=False, N=N, M1=M1, beta=beta, gamma=gamma)

        # Set up inverter (log-log for stability)
        bounds = (1e7, 1e16)  # [M_sun] Bounds cover all relevant halo masses
        inverter = NumericalInverter(
            forward, bounds, input_log=True, output_log=True)

        # Calculate mean halo mass (from numerical inverter)
        M_vir = inverter(M_disc)

        # Apply reverse scatter
        if scatter:
            # Calculate local slope at the retrieved halo mass
            slope = MassRelations.get_log_shmr_slope(
                M_vir, M1, beta, gamma)

            # Scatter_X = Scatter_Y / |slope|
            sigma_log_M_vir = sigma_log_M_disc / np.abs(slope)

            # Sample noise in log10(M_vir) space
            noise = np.random.normal(
                loc=0.0, scale=sigma_log_M_vir, size=M_vir.shape)

            # M_final = 10^(log(M_mean) + noise) = M_mean * 10^noise
            return M_vir * 10**noise

        return M_vir

    @staticmethod
    def pdf_schechter_log(
        log_M_disc,
        log_M_star=10.70, alpha=-1.37, phi_star=0.98e-12
    ):
        """
        Single Schechter function (logarithmic form).

            Phi(log M_disc) dlog M_disc = ln(10) * phi_star 
                * (M_disc/M_star)^(alpha+1) * exp(-M_disc/M_star) dlog M_disc

        Parameters
        ----------
        log_M_disc                  : array_like    # [log(M_sun)]
        log_M_star, alpha, phi_star : float         # [log(M_sun), 1, dex^-1 kpc^-3]
            Fitting parameters for the Schechter function. Default values are 
            from the disk dominated row in Table 3 of Kelvin et. al (2014).

        Returns
        -------
        probability_density         : np.ndarray    # [dex^-1 kpc^-3]
            Unnormalised probability density function, evaluated at log_M_disc.
        """
        log_M_disc = np.asarray(log_M_disc)
        ratio = 10**(log_M_disc - log_M_star)  # ratio = M_disc / M_star
        return np.log(10) * phi_star * ratio**(alpha + 1) * np.exp(-ratio)

    @staticmethod
    def pdf_double_schechter_log(
        log_M_disc,
        log_M_star=10.64, alpha1=-0.43, alpha2=-1.50, phi1=4.18e-12, phi2=0.74e-12
    ):
        """
        Double Schechter function (logarithmic form).

            Phi_double(log M_disc) dlog M_disc = (Phi_1 + Phi_2) dlog M_disc

        Parameters
        ----------
        log_M_disc                              : array_like    # [log(M_sun)]
        log_M_star, alpha1, alpha2, phi1, phi2  : float         # [log(M_sun), 1, 1, dex^-1 kpc^-3]
            Fitting parameters for the double Schechter function. 
            Default values are from Table 3 of Kelvin et. al (2014).

        Returns
        -------
        probability_density                     : np.ndarray    # [dex^-1 kpc^-3]
            Unnormalised probability density function, evaluated at log_M_disc.
        """
        log_M_disc = np.asarray(log_M_disc)
        ratio = 10**(log_M_disc - log_M_star)  # ratio = M_disc / M_star

        term1 = phi1 * ratio**(alpha1 + 1)
        term2 = phi2 * ratio**(alpha2 + 1)

        return np.log(10) * (term1 + term2) * np.exp(-ratio)

    @staticmethod
    def pdf_bounded_log_normal(log_M_disc, mu, sigma,
                               lower_cutoff=-np.inf, upper_cutoff=np.inf):
        """
        Ad-hoc log-normal pdf with sharp upper/lower cutoffs.

        Parameters
        ----------
        log_M_disc          : array_like    # [log(M_sun)]
        mu                  : float         # [log(M_sun)]
            Mean of the log-normal distribution in log-space.
        sigma               : float         # [dex]
            Standard deviation of the log-normal distribution in log-space.
        lower_cutoff        : float         # [log(M_sun)]
            Lower cutoff in log-space. PDF is zero for log_M_disc < lower_cutoff.
        upper_cutoff        : float         # [log(M_sun)]
            Upper cutoff in log-space. PDF is zero for log_M_disc > upper_cutoff.

        Returns
        -------
        probability_density : np.ndarray    # [dex^-1 kpc^-3]
            Unnormalised probability density function, evaluated at log_M_disc.
        """
        log_M_disc = np.asarray(log_M_disc)
        # log-normal distribution
        pdf = np.exp(-0.5 * ((log_M_disc - mu) / sigma)**2)

        # Hard cutoff: set probability to 0 where log_M_disc is outside the specified bounds
        pdf[log_M_disc < lower_cutoff] = 0.0
        pdf[log_M_disc > upper_cutoff] = 0.0

        return pdf


class StellarRelations:
    """
    Stateless physics engine for galactic stellar disc relations.
    Implements Shen et al. (2003) and Bershady et al. (2010) sampled relations,
    and other exact relations.

    All functions accept array-like inputs and return numpy arrays.
    """

    @staticmethod
    def get_half_light_radius_scatter(stellar_mass,
                                      M0=3.98e10, sigma1=0.47, sigma2=0.34):
        """Calculate the mass-dependent log-normal scatter in half-light 
        radius (R_hl) from stellar mass, sigma_ln_R_hl.

        Relation: Shen et al. (2003) Eq. 19.
            sigma_ln_R_hl = sigma2 + (sigma1 - sigma2) / (1 + (M/M0)^2)
        """
        M_disc = np.asarray(stellar_mass)
        return sigma2 + (sigma1 - sigma2) / (1 + (M_disc / M0)**2)

    @staticmethod
    def get_half_light_radius(
        stellar_mass,
        scatter=False,
        alpha=0.14, beta=0.39, gamma=0.1, M0=3.98e10,
        sigma1=0.47, sigma2=0.34
    ):
        """
        Calculate half-light radius from stellar mass.

        Relation: Shen et al. (2003) Eq. 18.
            R_hl = gamma * M_disc^alpha * (1 + M_disc/M0)^(beta-alpha)
        Scatter:  Shen et al. (2003) Eq. 19 (Mass-dependent, log-normal).

        Parameters
        ----------
        stellar_mass            : array_like    # [M_sun]
        scatter                 : bool  
            If True, applies intrinsic scatter to ln(R_hl)
        gamma, alpha, beta, M0  : float         # [1, 1, 1, M_sun]
            Fitting parameters for the size-mass relation (Table 1, Shen+03).
        sigma1, sigma2          : float         # [1, 1]
            Fitting parameters for the scatter (Table 1, Shen+03).

        Returns
        -------
        half_light_radius       : np.ndarray    # [kpc]
        """
        M_disc = np.asarray(stellar_mass)

        # Calculate mean relation
        R_hl = gamma * M_disc**alpha * (1 + M_disc / M0)**(beta - alpha)

        if scatter:
            # Calculate mass-dependent scatter width sigma_lnR
            # Note: Shen et al. define this as scatter in natural log space
            sigma_ln_R_hl = StellarRelations.get_half_light_radius_scatter(
                M_disc, M0, sigma1, sigma2)

            # Sample noise: N(0, sigma_ln_R)
            noise = np.random.normal(
                loc=0.0, scale=sigma_ln_R_hl, size=M_disc.shape)

            # Apply to mean (R = exp(ln(R_mean) + noise) = R_mean * exp(noise))
            return R_hl * np.exp(noise)

        return R_hl

    @staticmethod
    def get_scale_radius(half_light_radius):
        """
        Convert half-light radius to stellar disc scale radius.

            R_disc = - R_hl (1 - W_{-1}(-1/(2e))) = 0.595824 * R_hl
        where W_{-1} is the Lambert W function (branch -1).

        Parameters
        ----------
        half_light_radius   : array_like    # [kpc]
            Half-light radius of the stellar disc.

        Returns
        -------
        scale_radius        : np.ndarray    # [kpc]
            Scale radius of the stellar disc.
        """
        return 0.595824 * np.asarray(half_light_radius)

    @staticmethod
    def get_oblateness(
        scale_radius,
        scatter=False,
        slope=0.367, intercept=0.708,
        sigma_log_q=0.095
    ):
        """
        Calculate disc oblateness (q = scale radius / scale height) from scale radius.

        Relation: Bershady et al. (2010) Eq. 1.
            log(q) = slope * log(R_disc) + intercept

        Parameters
        ----------
        scale_radius        : array_like    # [kpc]
            Disc scale radius (R_disc) in kpc.
        scatter             : bool 
            If True, applies intrinsic scatter.
        slope, intercept    : float         # [1, 1]
            Fitting parameters for the log(q)-log(R_disc) relation (Bershady+10).
        sigma_log_q         : float         # [dex]
            Scatter in log10(q) relation.

        Returns
        -------
        oblateness          : np.ndarray
            Dimensionless ratio q = scale radius / scale height.
        """
        R_disc = np.asarray(scale_radius)

        # Calculate mean relation
        log_q = slope * np.log10(R_disc) + intercept

        if scatter:
            # Apply log-normal scatter
            noise = np.random.normal(
                loc=0.0, scale=sigma_log_q, size=R_disc.shape)
            log_q = log_q + noise

        # Convert to linear space
        return 10**log_q

    @staticmethod
    def get_scale_height(scale_radius, oblateness):
        """
        Calculate disc scale height.

            scale_height = scale_radius / oblateness

        Parameters
        ----------
        scale_radius    : array_like    # [kpc]
            Disc scale radius, R_disc.
        oblateness      : array_like    # [1]
            Dimensionless ratio q = scale radius / scale height.

        Returns
        -------
        scale_height    : np.ndarray    # [kpc]
            Disc scale height, z_disc.
        """
        return np.asarray(scale_radius) / np.asarray(oblateness)

    @staticmethod
    def get_density_normalisation(stellar_mass, scale_radius):
        """
        Calculate central surface density.

            M_disc = 2 * pi * Sigma_disc * R_disc^2

        Parameters
        ----------
        stellar_mass            : array_like    # [M_sun]
            Total stellar mass of the disc.
        scale_radius            : array_like    # [kpc]
            Disc scale radius, R_disc.

        Returns
        -------
        density_normalisation   : np.ndarray    # [M_sun / kpc^2]
            Central surface density, Sigma_disc.
        """
        denominator = 2 * np.pi * np.asarray(scale_radius)**2
        return np.asarray(stellar_mass) / denominator


class NFWHaloRelations:
    """
    Stateless physics engine for NFW dark matter halo relations.
    Implements Dutton & Maccio (2014) concentration-halo mass relation
    and NFW geometric definitions.

    All functions accept array-like inputs and return numpy arrays.
    """

    @staticmethod
    def get_concentration(
        virial_mass,
        h,
        scatter=False,
        slope=-0.101, intercept=0.905,
        sigma_log_c_vir=0.11
    ):
        """
        Calculate NFW concentration from virial mass.

        Relation: Dutton & Maccio (2014) Eq. 8.
            log c_vir = intercept - slope * log(M_vir / (1e12 h^-1))

        Parameters
        ----------
        virial_mass       : array_like    # [M_sun]
        h               : float         # [1]
            Hubble parameter (H0 = 100h km/s/Mpc). E.g. h=0.7.
        scatter         : bool
            If True, applies intrinsic log-normal scatter.
        slope, intercept: float         # [1, 1]
            Fitting parameters for log(c_vir)-log(M_vir) relation.
        sigma_log_c_vir     : float         # [dex]
            Scatter in log10(c_vir).

        Returns
        -------
        concentration   : np.ndarray    # [1]
        """
        M_vir = np.asarray(virial_mass)

        # Calculate mean relation
        M_pivot = 1e12 / h
        log_c_vir = intercept + slope * np.log10(M_vir / M_pivot)

        if scatter:
            # Apply log-normal scatter
            noise = np.random.normal(
                loc=0.0, scale=sigma_log_c_vir, size=M_vir.shape)
            log_c_vir = log_c_vir + noise

        # Convert to linear space
        return 10**log_c_vir

    @staticmethod
    def get_virial_radius(virial_mass, rho_crit, Delta_vir):
        """
        Calculate virial radius from virial mass.

            M_vir = 4/3 * pi * Delta * rho_crit * r_vir^3
         => r_vir = (3 M_vir / (4 pi Delta rho_crit))^(1/3)
        Parameters
        ----------
        virial_mass       : array_like    # [M_sun]
        rho_crit        : float         # [M_sun / kpc^3]
            Critical density parameter.
        Delta_vir       : float         # [1] 
            Virial critical overdensity (e.g. Delta_200c = 200)

        Returns
        -------
        virial_radius   : np.ndarray    # [kpc]
        """
        M_vir = np.asarray(virial_mass)

        denominator = 4 * np.pi * Delta_vir * rho_crit
        prefactor = 3 / denominator

        return (prefactor * M_vir)**(1/3)

    @staticmethod
    def get_scale_radius(virial_radius, concentration):
        """
        Calculate NFW scale radius.

            r_nfw = r_vir / c_vir

        Parameters
        ----------
        virial_radius   : array_like    # [kpc]
            Virial radius of the halo.
        concentration   : array_like    # [1]
            NFW concentration parameter.

        Returns
        -------
        scale_radius    : np.ndarray    # [kpc]
            NFW scale radius, r_nfw.
        """
        return np.asarray(virial_radius) / np.asarray(concentration)

    @staticmethod
    def get_density_normalisation(virial_mass, scale_radius, concentration):
        """
        Calculate NFW density normalisation (rho_nfw).
        Derived by evaluating enclosed mass function at the virial radius.

            M_vir = 4 * pi * rho_nfw * r_nfw^3 * [ln(1+c_vir) - c_vir/(1+c_vir)]

        Parameters
        ----------
        virial_mass               : array_like    # [M_sun]
            Virial mass of the halo
        scale_radius            : array_like    # [kpc]
            NFW scale radius, r_nfw.
        concentration           : array_like    # [1]
            NFW concentration parameter, c_vir.

        Returns
        -------
        density_normalisation   : np.ndarray    # [M_sun / kpc^3]
            NFW density normalisation, rho_nfw.
        """
        M_vir = np.asarray(virial_mass)
        r_nfw = np.asarray(scale_radius)
        c_vir = np.asarray(concentration)

        # Calculate function of concentration: f(c) = ln(1+c) - c/(1+c)
        f_c_vir = np.log(1 + c_vir) - (c_vir / (1 + c_vir))

        # rho_nfw = M_vir / (4 * pi * r_nfw^3 * f_c_vir)
        numerator = M_vir
        denominator = 4 * np.pi * r_nfw**3 * f_c_vir

        return numerator / denominator
