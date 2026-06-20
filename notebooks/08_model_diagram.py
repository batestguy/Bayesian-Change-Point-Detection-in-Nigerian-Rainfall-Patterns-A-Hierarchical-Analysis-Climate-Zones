"""
Generate Kruschke-style hierarchical model diagram using daft.
Produces fig_model_diagram.pdf for the IEEE paper methodology section.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import daft

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "text.usetex": False,
    "font.size": 10,
})

pgm = daft.PGM(
    observed_style="shaded",
    dpi=300,
    node_unit=1.6,
    node_ec="black",
    directed=True,
    line_width=1.0,
)

# ── Hyperpriors (above plate) ──
pgm.add_node("tau_mu", r"$\tau_\mu$", 3, 7, scale=1.1,
             plot_params={"facecolor": "white"})
pgm.add_node("tau_sigma", r"$\tau_\sigma$", 5, 7, scale=1.1,
             plot_params={"facecolor": "white"})
pgm.add_node("kappa", r"$\kappa$", 8, 7, scale=1.1,
             plot_params={"facecolor": "white"})

# ── Zone offset ──
pgm.add_node("tau_offset", r"$\delta_z$", 4, 5.2, scale=1.1,
             plot_params={"facecolor": "white"})

# ── Zone parameters ──
pgm.add_node("mu_before", r"$\mu_z^{\rm bef}$", 1.2, 3.5, scale=1.1,
             plot_params={"facecolor": "white"})
pgm.add_node("tau_z", r"$\tau_z$", 4, 3.5, scale=1.1,
             plot_params={"facecolor": "#D0D0D0"})
pgm.add_node("mu_after", r"$\mu_z^{\rm aft}$", 6.8, 3.5, scale=1.1,
             plot_params={"facecolor": "white"})
pgm.add_node("sigma_z", r"$\sigma_z$", 9, 3.5, scale=1.1,
             plot_params={"facecolor": "white"})

# ── Observed ──
pgm.add_node("y", r"$y_{z,t}$", 4, 1.5, scale=1.3,
             observed=True)

# ── Edges ──
pgm.add_edge("tau_mu", "tau_z")
pgm.add_edge("tau_sigma", "tau_z")
pgm.add_edge("tau_offset", "tau_z")
pgm.add_edge("mu_before", "y")
pgm.add_edge("mu_after", "y")
pgm.add_edge("tau_z", "y")
pgm.add_edge("sigma_z", "y")
pgm.add_edge("kappa", "y")

# ── Plates ──
pgm.add_plate([0.2, 0.5, 9.8, 5.5], label=r"zone $z = 1, \ldots, Z$",
              position="bottom left", shift=-0.12,
              rect_params={"linewidth": 1.5})
pgm.add_plate([2.8, 0.6, 2.5, 1.8], label=r"$t = 1 \ldots T$",
              position="bottom right", shift=-0.1,
              rect_params={"linewidth": 1.0, "linestyle": "--"})

# ── Distribution annotations via pgm.add_text ──
pgm.add_text(3, 7.65, r"$\mathcal{N}(30,10)$", fontsize=7)
pgm.add_text(5, 7.65, r"$\mathrm{HalfN}(5)$", fontsize=7)
pgm.add_text(8, 7.65, r"$\mathrm{HalfN}(5)$", fontsize=7)
pgm.add_text(4, 5.8, r"$\mathcal{N}(0,1)$", fontsize=7)
pgm.add_text(1.2, 4.15, r"$\mathcal{N}(\bar{y}_z, 200)$", fontsize=7)
pgm.add_text(6.8, 4.15, r"$\mathcal{N}(\bar{y}_z, 200)$", fontsize=7)
pgm.add_text(9, 4.15, r"$\mathrm{HalfN}(s_z)$", fontsize=7)
pgm.add_text(4, 2.85, r"$= \tau_\mu + \tau_\sigma \cdot \delta_z$", fontsize=6.5)
pgm.add_text(4, 0.85, r"$\mathcal{N}(\mu_t, \sigma_z)$", fontsize=7)
pgm.add_text(4, 0.35, r"$\mu_t = \mu_z^{\rm bef}(1\!-\!w) + \mu_z^{\rm aft}w$", fontsize=5.5)
pgm.add_text(4, -0.05, r"$w = \sigma(\kappa \cdot (t - \tau_z))$", fontsize=5.5)

# ── Render ──
pgm.render()
pgm.savefig("figures/fig_model_diagram.pdf", dpi=300)
pgm.savefig("figures/fig_model_diagram.png", dpi=300)
print("Saved figures/fig_model_diagram.pdf and .png")
