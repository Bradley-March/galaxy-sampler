import numpy as np
import matplotlib.pyplot as plt
import corner
import inspect
from relations import MassRelations, StellarRelations, NFWHaloRelations
from cosmology import Cosmology


def plot_population_corner(population, bins=30, title="Galaxy Population",
                           filename=None, cosmo=None,
                           smf_pdf=None, **smf_kwargs):
    """
    Generates a corner plot for each of the sampled galactic parameters.
    Also plots the underlying mean relations (and +-1 sigma contours) 
    on top of the relevant panels for visual comparison.
    """
    if cosmo is None:
        cosmo = Cosmology()

    # Define what to plot (label, log data drray)
    data_map = [
        (r"$\log M_{vir}$",     population.halo.log.mass),
        (r"$\log c_{vir}$",     population.halo.log.concentration),
        (r"$\log M_{disc}$",    population.disc.log.mass),
        (r"$\log R_{hl}$",      population.disc.log.half_light_radius),
        (r"$\log q$",           population.disc.log.oblateness),
    ]
    labels = [item[0] for item in data_map]
    data_stack = np.column_stack([item[1] for item in data_map])

    # Filter out NaNs (shouldn't be any but worthwhile safety check)
    mask = np.isfinite(data_stack).all(axis=1)
    clean_data = data_stack[mask]
    if np.sum(~mask) > 0:
        print(f"Warning: {np.sum(~mask)} galaxies had NaN/Inf values",
              "and were excluded from the plot.")

    # Create corner plot
    fig = corner.corner(
        clean_data,
        bins=bins,
        labels=labels,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        color="#1f77b4",
        plot_datapoints=True,
        plot_density=False,
        fill_contours=True,
        levels=[0.68, 0.95]  # [1, 2] sigma contours
    )

    # Overplot "True" mean relations on the relevant panels
    axes = np.array(fig.axes).reshape(len(labels), len(labels))

    # --- SMF (log M_disc) ---
    # We only plot the SMF if the user provided a PDF function (e.g., Schechter)
    if smf_pdf is not None:
        ax_smf = axes[2, 2]  # Row 2 (M_disc), col 2 (M_disc)
        # Generate a smooth grid for the line
        min_x, max_x = ax_smf.get_xlim()
        log_M_disc_grid = np.linspace(min_x, max_x, 100)
        # Calculate bin width for normalisation
        dx = (max_x - min_x) / log_M_disc_grid.size
        dx_hist = (max_x - min_x) / bins
        # Evaluate the SMF on the grid
        smf_values = smf_pdf(log_M_disc_grid, **smf_kwargs)
        # Normalise smf to pdf (ensure area under curve = 1)
        norm_factor = np.sum(smf_values) * dx
        smf_values_scaled = smf_values / norm_factor
        # Scale to match the histogram density
        smf_values_scaled = smf_values_scaled * dx_hist * population.size
        # Plot the SMF line
        ax_smf.plot(log_M_disc_grid, smf_values_scaled, color='orange', lw=2)

    # --- SHMR (log M_vir vs log M_disc) ---
    ax_shmr = axes[2, 0]  # Row 2 (M_disc), col 0 (M_vir)
    # Generate a smooth grid for the line
    log_M_vir_grid = np.linspace(*ax_shmr.get_xlim(), 100)
    log_M_disc_mean = np.log10(MassRelations.get_stellar_mass_from_virial_mass(
        10**log_M_vir_grid, scatter=False))
    # Plot mean relation
    ax_shmr.plot(log_M_vir_grid, log_M_disc_mean, color='orange', lw=2)
    # Plot +/- 1 sigma scatter bounds

    shmr_signature = inspect.signature(
        # fetch the function signature to inspect the default scatter value
        MassRelations.get_stellar_mass_from_virial_mass)
    shmr_scatter = shmr_signature.parameters['sigma_log_M_disc'].default
    ax_shmr.plot(log_M_vir_grid, log_M_disc_mean + shmr_scatter,
                 'k--', alpha=0.5)
    ax_shmr.plot(log_M_vir_grid, log_M_disc_mean - shmr_scatter,
                 'k--', alpha=0.5)

    # --- NFW Concentration-Mass (log c_vir vs log M_vir) ---
    ax_cm = axes[1, 0]  # Row 1 (c_{vir}), col 0 (M_vir)
    # Generate a smooth grid for the line (same mh_grid as SHMR)
    log_c_vir_mean = np.log10(NFWHaloRelations.get_concentration(
        10**log_M_vir_grid, h=cosmo.h, scatter=False))
    # Plot Mean
    ax_cm.plot(log_M_vir_grid, log_c_vir_mean, color='orange', lw=2)
    # Plot +/- 1 sigma scatter bounds (Assuming 0.1 dex for visual simplicity)
    cm_signature = inspect.signature(NFWHaloRelations.get_concentration)
    cm_scatter = cm_signature.parameters['sigma_log_c_vir'].default
    ax_cm.plot(log_M_vir_grid, log_c_vir_mean + cm_scatter, 'k--', alpha=0.5)
    ax_cm.plot(log_M_vir_grid, log_c_vir_mean - cm_scatter, 'k--', alpha=0.5)

    # --- Mass-Size (log M_disc vs log R_hl) ---
    ax_ms = axes[3, 2]  # Row 3 (R_hl), col 1 (M_disc)
    # Generate smooth grid for the line
    log_M_disc_grid = np.linspace(*ax_ms.get_xlim(), 100)
    log_R_hl_mean = np.log10(StellarRelations.get_half_light_radius(
        10**log_M_disc_grid, scatter=False))
    # Plot mean relation
    ax_ms.plot(log_M_disc_grid, log_R_hl_mean, color='orange', lw=2)
    # Plot +/- 1 sigma scatter bounds (using mass-dependent scatter from Shen+03)
    ln_R_hl_scatter = StellarRelations.get_half_light_radius_scatter(
        10**log_M_disc_grid)
    log_R_hl_scatter = ln_R_hl_scatter / np.log(10)  # convert to log10 space
    ax_ms.plot(log_M_disc_grid, log_R_hl_mean +
               log_R_hl_scatter, 'k--', alpha=0.5)
    ax_ms.plot(log_M_disc_grid, log_R_hl_mean -
               log_R_hl_scatter, 'k--', alpha=0.5)

    # --- Oblateness (log q vs log R_hl) ---
    ax_obl = axes[4, 3]  # Row 4 (q), col 3 (R_hl)
    # Generate a smooth grid for the line
    log_R_hl_grid = np.linspace(*ax_obl.get_xlim(), 100)
    # convert half-light radius to scale radius to calculate oblateness
    R_disc_grid = StellarRelations.get_scale_radius(10**log_R_hl_grid)
    # Get mean oblateness from scale radius
    log_q_mean = np.log10(
        StellarRelations.get_oblateness(R_disc_grid, scatter=False))
    # Plot mean relation
    ax_obl.plot(log_R_hl_grid, log_q_mean, color='orange', lw=2)
    # plot +/- 1 sigma scatter bounds
    obl_signature = inspect.signature(StellarRelations.get_oblateness)
    obl_scatter = obl_signature.parameters['sigma_log_q'].default
    ax_obl.plot(log_R_hl_grid, log_q_mean + obl_scatter, 'k--', alpha=0.5)
    ax_obl.plot(log_R_hl_grid, log_q_mean - obl_scatter, 'k--', alpha=0.5)

    # Final formatting
    fig.suptitle(title, fontsize=16)

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()


# %% Example Usage
if __name__ == "__main__":
    from pipeline import GalaxyPipeline
    from time import time

    # Set rng seed for reproducibility
    np.random.seed(42)

    ### Choose a custom SMF PDF and parameters (optional) ###
    # log normal example
    smf_pdf = MassRelations.pdf_bounded_log_normal
    smf_kwargs = {"mu": 10.6, "sigma": 0.35,
                  "lower_cutoff": 8.0, "upper_cutoff": 11.0}
    # Schechter example
    smf_pdf = MassRelations.pdf_schechter_log
    smf_kwargs = {}
    # Double Schechter example
    smf_pdf = MassRelations.pdf_double_schechter_log
    smf_kwargs = {}
    # Also possible to use a custom user-defined PDF function, e.g. power law.

    ### Run the pipeline to generate a population ###
    t1 = time()
    pl = GalaxyPipeline()
    galpop = pl.generate_population(
        n_samples=100_000, scatter=True, smf_pdf=smf_pdf, **smf_kwargs)
    t2 = time()
    print(f"Generated population of {galpop.size} in {t2 - t1:.2f} seconds.")

    ### Plot the population in a corner plot with mean relations overplotted ###
    plot_population_corner(galpop, bins=30,
                           title=f"Example Galaxy Population with n={galpop.size}",
                           smf_pdf=smf_pdf, **smf_kwargs)
    t3 = time()
    print(f"Plotted population in {t3 - t2:.2f} seconds.")
