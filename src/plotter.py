import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import unicodedata
import scipy.cluster.hierarchy as sch
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects
import seaborn as sns
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from src.model.simulation import Simulation
matplotlib.use('agg')


class Plotter:
    def __init__(self, scaling_object, countries_to_plot,
                 base_on: str, base_frac: int, base_r0: int,
                 coordinate_loader, save_dir="figures"):
        """
        scaling_object: instance of ContactMatrixScaling
        countries_to_plot: list of countries ["Kenya", "Angola", "South Africa"]
        save_dir: folder to save the figures
        """

        self.scaling_object = scaling_object
        self.countries = countries_to_plot
        self.coordinate_loader = coordinate_loader
        self.base_on = base_on
        self.base_frac = base_frac
        self.base_r0 = base_r0
        self.save_dir = save_dir

        self.data = scaling_object.data  # reuse the same DataLoader instance
        self.setting_contacts = scaling_object.setting_contacts  # reuse scaled contacts

        os.makedirs(self.save_dir, exist_ok=True)

    def plot_distance_matrix(self, cluster_obj):
        """Plot raw distances between countries before reordering."""

        distance_df, _ = cluster_obj.get_distance_matrix()
        countries = cluster_obj.country_names

        fig, ax = plt.subplots(figsize=(34, 34), dpi=300)
        cax = ax.imshow(distance_df, cmap="jet", vmin=0)

        # Set ticks and labels only on left and bottom
        ax.set_xticks(np.arange(len(countries)))
        ax.set_xticklabels(countries, rotation=90, fontsize=25)
        ax.xaxis.tick_bottom()

        ax.set_yticks(np.arange(len(countries)))
        ax.set_yticklabels(countries, fontsize=25)
        ax.yaxis.tick_left()

        # Improve colorbar
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=40)
        # Clean style
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "distance_matrix_raw.pdf"))
        plt.close()

    def plot_ordered_distance_matrix(self, dt, columns):
        """Plot reordered distance matrix with better visuals."""
        fig, ax = plt.subplots(figsize=(34, 34), dpi=300)
        cax = ax.imshow(dt, cmap="magma", vmin=0)

        # Axis ticks only on bottom and left
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=90, fontsize=25)
        ax.xaxis.tick_bottom()

        ax.set_yticks(np.arange(len(columns)))
        ax.set_yticklabels(columns, fontsize=25)
        ax.yaxis.tick_left()

        # Colorbar styling
        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=40)

        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "distance_matrix_ordered.pdf"))
        plt.close()

    def plot_africa_map_with_epidemic_indicators(self, leaf_colors=None):
        def scale_radii(values, min_radius=0.1, max_radius=0.5):
            values = np.array(values, dtype=float)
            values = np.clip(values, 1, None)
            log_vals = np.log10(values)
            norm = (log_vals - log_vals.min()) / (log_vals.max() - log_vals.min() + 1e-5)
            return min_radius + norm * (max_radius - min_radius)

        alias_map = {
            "DR Congo": "Democratic Republic of the Congo",
            "Congo": "Republic of the Congo",
            "Cape Verde": "Cabo Verde",
            "Swaziland": "Eswatini",
            "Sao Tome and Principe": "São Tomé and Principe",
            "Tanzania": "United Republic of Tanzania"
        }

        def normalize(name):
            return unicodedata.normalize('NFKD',
                                         name).encode('ASCII', 'ignore').decode().lower()

        africa_map = self.coordinate_loader.africa_map
        country_coords = self.coordinate_loader.get_all_coordinates()

        map_names = set(africa_map["name"])
        normalized_map_names = {normalize(name): name for name in map_names}

        all_country_names = self.scaling_object.country_names
        mapped_names = {}
        for name in all_country_names:
            aliased = alias_map.get(name, name)
            normed = normalize(aliased)
            if normed in normalized_map_names:
                mapped_names[name] = normalized_map_names[normed]
            else:
                print(f"❌ No match found for: {name} → {aliased}")

        inverse_name_mapping = {v: k for k, v in mapped_names.items()}

        relevant_names = set(mapped_names.values()).union({"Western Sahara"})
        africa_map = africa_map[africa_map["name"].isin(relevant_names)].copy()
        africa_map["label_name"] = africa_map["name"].map(lambda x:
                                                          inverse_name_mapping.get(x, x))
        if leaf_colors:
            africa_map["fill_color"] = africa_map[
                "label_name"].map(lambda name: leaf_colors.get(name, "lightgray"))
        else:
            africa_map["fill_color"] = "lightgray"

        # Override fill color for Western Sahara explicitly
        africa_map.loc[africa_map["name"] == "Western Sahara", "fill_color"] = "white"

        fig, ax = plt.subplots(figsize=(16, 18))
        fig.patch.set_facecolor('white')

        africa_map.plot(ax=ax, facecolor=africa_map["fill_color"], edgecolor='black',
                        linewidth=0.7, alpha=0.5)

        population_dict = self.scaling_object.population
        icu_dict, deaths_dict = {}, {}
        for country in all_country_names:
            if country not in self.setting_contacts:
                continue
            age_vector = self.setting_contacts[country]["age_vector"]
            sim = Simulation(
                data=self.scaling_object.data,
                contact_matrix=self.setting_contacts[country]["contact_full_unscaled"],
                age_vector=age_vector,
                susceptibility=np.ones(16),
                base_r0=self.base_r0,
                base_frac=self.base_frac,
                country=country,
                base_on=self.base_on
            )
            t = np.arange(0, 600, 1)
            sol = sim.model.get_solution(
                init_values=sim.model.get_initial_values(),
                t=t, parameters=sim.params,
                cm=self.setting_contacts[country]["contact_full_unscaled"]
            )
            icu_dict[country] = sim.model.get_icu(sol).max()
            deaths_dict[country] = sim.model.get_deaths(sol)[-1]

        specs = {
            "Population": {
                "data": population_dict,
                "bins": [0, 10e6, 20e6, 50e6, 100e6, 1e9],
                "color": "red",
                "offset": (-0.7, 0.7)
            },
            "ICU Max": {
                "data": icu_dict,
                "bins": [0, 500, 1000, 5000, 10000, 1e6],
                "color": "cyan",
                "offset": (0, 0)
            },
            "Total Deaths": {
                "data": deaths_dict,
                "bins": [0, 1000, 5000, 10000, 50000, 1e6],
                "color": "darkblue",
                "offset": (0.7, 0.7)
            }
        }

        legend_blocks = []
        for key, spec in specs.items():
            data, color, offset, bins = spec["data"], spec["color"], spec["offset"], \
                                        spec["bins"]
            bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
            radii = scale_radii(bin_centers)

            if key == "Population":
                bin_labels = ["0–10M", "10M–20M", "20M–50M", "50M–100M", "≥100M"]
            else:
                bin_labels = [
                    f"{int(bins[i]):,}–{int(bins[i + 1]):,}" if
                    i < len(bins) - 2 else f"≥{int(bins[i]):,}"
                    for i in range(len(bins) - 1)
                ]

            bin_indices = {country: np.digitize(val, bins, right=False) -
                                    1 for country, val in data.items()}

            for sim_country in all_country_names:
                geojson_name = mapped_names.get(sim_country)
                if geojson_name is None:
                    continue
                coords = country_coords.get(geojson_name)
                if coords is None:
                    continue
                lat, lon = coords
                idx = bin_indices.get(sim_country, -1)
                if 0 <= idx < len(radii):
                    r = radii[idx]
                    dx, dy = offset
                    ax.add_patch(plt.Circle((lon + dx, lat + dy), radius=r,
                                            color=color, alpha=0.4,
                                            ec='black', lw=0.3))

            handles = [
                Line2D([], [], marker='o', linestyle='None',
                       markerfacecolor=color, markeredgecolor='black', alpha=0.8,
                       markersize=r * 30, label=label)
                for r, label in zip(radii, bin_labels)
            ]
            legend_blocks.append((key, handles))

        for country, (lat, lon) in country_coords.items():
            label_name = inverse_name_mapping.get(country, country)
            ax.text(lon, lat - 1.0, label_name, fontsize=9,
                    ha='center', va='center', color='black',
                    path_effects=[patheffects.Stroke(linewidth=3, foreground='white'),
                                  patheffects.Normal()])

        legend_dict = {title: handles for title, handles in legend_blocks}
        if leaf_colors:
            color_to_cluster = {}
            for label, color in leaf_colors.items():
                if color not in color_to_cluster:
                    color_to_cluster[color] = len(color_to_cluster) + 1

            cluster_handles = [
                mpatches.Patch(facecolor=color, edgecolor='black',
                               label=f"Cluster {cid}")
                for color, cid in color_to_cluster.items()
            ]

            # Add white patch for "No Data" (Western Sahara)
            cluster_handles.append(
                mpatches.Patch(facecolor="white", edgecolor='black',
                               label="No Contact Data")
            )

            legend_dict["Clusters"] = cluster_handles

        legend_grid_positions = {
            "Population": (0.18, 0.01),
            "Clusters": (0.18, 0.25),
            "ICU Max": (0.01, 0.01),
            "Total Deaths": (0.01, 0.25)
        }

        for title, (y, x) in legend_grid_positions.items():
            handles = legend_dict.get(title)
            if handles:
                leg = ax.legend(handles=handles, title=title,
                                loc='lower left', bbox_to_anchor=(x, y),
                                fontsize=9, title_fontsize=10,
                                frameon=True, borderpad=1, handletextpad=1.5,
                                labelspacing=1.2)
                ax.add_artist(leg)

        ax.set_xlim(-25, 60)
        ax.set_ylim(-40, 40)
        ax.set_xlabel("Longitude", fontsize=20)
        ax.set_ylabel("Latitude", fontsize=20)
        ax.tick_params(axis='both', labelsize=12, width=2, length=6, color="black")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.5)
        ax.grid(False)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "african_map.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_dendrogram_with_threshold(self, res, threshold, labels):
        """
        Plot a dendrogram and return a mapping of leaf labels to their visual colors.
        """
        fig, ax = plt.subplots(figsize=(18, 10), dpi=300)

        cluster_colors = [
            "#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4",
            "#fed9a6", "#ffff99", "#a6cee3", "#fb8072"
        ]
        # Recommended for dendrogram labels (stronger but still tasteful)
        cluster_colors = [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3"
        ]

        sch.set_link_color_palette(cluster_colors)

        dendro = sch.dendrogram(
            res,
            color_threshold=threshold,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=14,
            above_threshold_color='dimgray',
            ax=ax
        )

        # Map each leaf label to its corresponding color
        leaf_colors = dict(zip(dendro['ivl'], dendro['leaves_color_list']))

        # Apply label coloring
        for lbl in ax.get_xmajorticklabels():
            label_text = lbl.get_text()
            if label_text in leaf_colors:
                lbl.set_color(leaf_colors[label_text])
            lbl.set_fontweight("medium")

        for line in ax.get_lines():
            if max(line.get_ydata()) > 0:
                line.set_linewidth(1.2)

        ax.set_ylabel('Cluster Distance', fontsize=22, fontweight="bold", color="black")
        ax.tick_params(axis='y', labelsize=18, width=2, length=6, colors="black")
        ax.tick_params(axis='x', labelsize=14, rotation=90, pad=6)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "dendrogram_with_threshold.pdf"),
                    bbox_inches="tight")
        plt.close()

        return leaf_colors

    def plot_selected_country_contact_matrices(self, data):
        labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                  "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
                  "70-74", "75+"]

        for country in self.countries:
            if country not in self.setting_contacts:
                print(f"Skipping {country}: no contact data.")
                continue

            # Unscaled and scaled matrix dictionaries
            contact_data = data.contact_data[country]
            scaled_data = self.setting_contacts[country]

            matrices_unscaled = {
                "Home": contact_data["home"],
                "School": contact_data["school"],
                "Work": contact_data["work"],
                "Other": contact_data["other"],
                "Full": contact_data["home"] + contact_data["school"] +
                        contact_data["work"] + contact_data["other"]
            }

            matrices_scaled = {
                "Home": scaled_data["contact_home"],
                "School": scaled_data["contact_school"],
                "Work": scaled_data["contact_work"],
                "Other": scaled_data["contact_other"],
                "Full": scaled_data["contact_full"]
            }

            output_dir = os.path.join("figures", "contact_matrices",
                                      country.replace(" ", "_"))
            os.makedirs(output_dir, exist_ok=True)

            # Compute separate colorbar ranges
            full_matrix_unscaled = matrices_unscaled["Full"]
            full_matrix_scaled = matrices_scaled["Full"]

            vmin_unscaled = max(full_matrix_unscaled[full_matrix_unscaled > 0].min(), 0.01)
            vmax_unscaled = full_matrix_unscaled.max()
            norm_unscaled = Normalize(vmin=vmin_unscaled, vmax=vmax_unscaled)

            vmin_scaled = max(full_matrix_scaled[full_matrix_scaled > 0].min(), 0.001)
            vmax_scaled = full_matrix_scaled.max()
            norm_scaled = Normalize(vmin=vmin_scaled, vmax=vmax_scaled)

            for scale_label, matrix_dict, norm in zip(
                    ["Unscaled", "Scaled"],
                    [matrices_unscaled, matrices_scaled],
                    [norm_unscaled, norm_scaled]):

                for name, matrix in matrix_dict.items():
                    df = pd.DataFrame(matrix, index=labels, columns=labels)

                    fig, ax = plt.subplots(figsize=(4, 4))
                    cmap = "Reds"
                    show_cbar = name == "Full"

                    sns.heatmap(df, cmap=cmap, norm=norm, linewidths=0.5,
                                square=True, cbar=False, ax=ax)

                    if show_cbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="3%", pad=0.2)
                        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array([])
                        cb = fig.colorbar(sm, cax=cax)
                        cb.ax.yaxis.set_label_position('right')
                        cb.ax.yaxis.set_ticks_position('right')
                        cb.ax.tick_params(labelsize=12, width=0.6, length=3)

                    # ax.set_title(f"{country}", fontsize=16, fontweight="bold", pad=10)
                    ax.set_title(f"{name}", fontsize=16, fontweight="bold", pad=10)
                    ax.set_xlabel("Respondent age", fontsize=10)
                    ax.set_ylabel("Contact age", fontsize=10)
                    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=8)
                    ax.set_yticklabels(labels, fontsize=8)
                    ax.tick_params(length=3, width=1)

                    # Hide top and right spines
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(True)
                    ax.spines["left"].set_linewidth(1)
                    ax.spines["left"].set_color("black")
                    ax.spines["bottom"].set_visible(True)
                    ax.spines["bottom"].set_linewidth(1)
                    ax.spines["bottom"].set_color("black")

                    ax.invert_yaxis()

                    plt.tight_layout()
                    filename = os.path.join(output_dir,
                                            f"{scale_label}_{name}_contacts.pdf")
                    plt.savefig(filename, format="pdf", bbox_inches="tight")
                    plt.close()

    def simulate_country(self, country):
        unscaled_contact = self.setting_contacts[country]["contact_full_unscaled"]
        scaled_contact = self.setting_contacts[country]["contact_full"]
        age_vector = self.setting_contacts[country]["age_vector"]
        susceptibility = np.ones(16)
        susceptibility[:4] = 0.5

        sim = Simulation(
            data=self.data,
            contact_matrix=scaled_contact,
            age_vector=age_vector,
            susceptibility=susceptibility,
            base_r0=self.base_r0,
            base_frac=self.base_frac,
            country=country,
            base_on=self.base_on
        )

        t = np.arange(0, 600, 1)
        sol = sim.model.get_solution(
            init_values=sim.model.get_initial_values(),
            t=t,
            parameters=sim.params,
            cm=scaled_contact
        )

        return sim, sol, t

    def plot_country(self, country):
        sim, sol, t = self.simulate_country(country)

        infected = sim.model.get_infected(sol)
        hospitalized = sim.model.get_hospitalized(sol)
        icu = sim.model.get_icu(sol)
        deaths = sim.model.get_deaths(sol)

        self.plot_hospital_icu(t, hospitalized, icu, country)
        self.plot_single_curve(t, infected,  "Prevalence", "#FC8D62", country)
        self.plot_single_curve(t, deaths, "Deaths", "green", country)

    def plot_deaths_vs_beta_scatter(self, leaf_colors: dict,
                                    filename="deaths_vs_beta_scatter.pdf"):
        beta_list = []
        deaths_list = []
        labels = []
        colors = []

        for country in self.scaling_object.country_names:
            if country not in self.setting_contacts:
                continue

            sim, sol, _ = self.simulate_country(country)
            beta = self.setting_contacts[country]["beta"]
            total_deaths = sim.model.get_deaths(sol)[-1]

            beta_list.append(beta)
            deaths_list.append(total_deaths)
            labels.append(country)
            colors.append(leaf_colors.get(country, "gray"))

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(beta_list, deaths_list, s=70,
                   c=colors, edgecolors='black', linewidths=0.4, alpha=0.85)

        # Label each point
        for x, y, label, color in zip(beta_list, deaths_list, labels, colors):
            ax.text(x + 0.0001, y, label, fontsize=9,
                    ha='left', va='center', weight='bold', color=color,
                    path_effects=[
                        patheffects.Stroke(linewidth=2.2, foreground='white'),
                        patheffects.Normal()
                    ])

        # Set tighter x-limits to exaggerate the β scale
        margin = 0.0007
        xmin = min(beta_list) - margin
        xmax = max(beta_list) + margin
        ax.set_xlim(xmin, xmax)

        ax.set_xlabel("β (Transmission Scaling Parameter)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Total Deaths", fontsize=16, fontweight="bold")

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='both', labelsize=13, width=2, length=6, colors='black')

        # Apply custom spine and tick styling
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(2.5)
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_position(("outward", 10))

        ax.minorticks_off()
        ax.tick_params(axis='both', which='major', direction='out',
                       length=8, width=3, colors='black',
                       bottom=True, top=False, left=True, right=False,
                       labelsize=15)

        save_path = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"scatter plot saved to: {save_path}")

    def plot_hospital_icu(self, t, hospitalized, icu, country):
        """Plot Hospitalized and ICU curves together."""
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t, hospitalized, color="#E7B800", linewidth=1.2, label="Hospitalized")
        ax.plot(t, icu, color="#D62F5E", linewidth=1.2, label="ICU")

        ax.set_xlim(0, 400)
        ax.set_ylim(0, hospitalized.max() * 1.1)
        ax.set_xlabel("Days since release", fontsize=12)
        ax.set_ylabel("Number of individuals", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

        # Clean style
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(False)
        ax.set_title(f"Hospital/ICU Needs - {country}",
                     fontsize=14, weight="bold", pad=10)
        ax.legend(fontsize=10, frameon=False, loc="upper right")

        # Save figure
        country_folder = os.path.join(self.save_dir, country, self.base_on)
        os.makedirs(country_folder, exist_ok=True)
        save_path = os.path.join(country_folder, f"hospital_icu_curve.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()

    def plot_single_curve(self, t, series, name, color, country):
        """Plot single epidemic outcome curve (Incidence or Deaths)."""
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t, series, color=color, linewidth=1.2)

        ax.set_xlim(0, 400)
        ax.set_ylim(0, series.max() * 1.1)
        ax.set_xlabel("Days since release", fontsize=12)
        ax.set_ylabel("Number of individuals", fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(False)
        ax.set_title(f"{name} - {country}", fontsize=14, weight="bold", pad=10)
        # Save figure
        country_folder = os.path.join(self.save_dir, country, self.base_on)
        os.makedirs(country_folder, exist_ok=True)
        save_path = os.path.join(country_folder, f"{name.lower()}_curve.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()

    def run(self):
        """Run plotting for all selected countries."""
        for country in self.countries:
            if country in self.setting_contacts:
                print(f"Plotting {country}...")
                self.plot_country(country)
            else:
                print(f"Warning: {country} not found in scaled contacts!")
