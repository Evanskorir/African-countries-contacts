from src.contact_mtx_manipulator import ContactMatrixScaling
from src.coordinatesloader import CoordinateLoader
from src.clustering import Hierarchical
from src.dataloader import DataLoader
from src.plotter import Plotter


class PipelineManager:
    def __init__(self):
        self.ContactMatrixScaling = ContactMatrixScaling
        self.CoordinateLoader = CoordinateLoader
        self.DataLoader = DataLoader
        self.Plotter = Plotter
        self.Hierarchical = Hierarchical

        # Configuration parameters
        self.base_r0 = 3.68
        self.base_on = "deaths"  # ("r0", "infected", "peak", "hospital", "icu")
        self.base_frac = 0.0005
        self.susc = 1.0
        self.threshold = 0.15
        self.countries_to_plot = ["Kenya", "Benin", "Namibia", "DR Congo", "Tanzania"]

    def run(self):
        data = self.DataLoader()
        scaling = self.ContactMatrixScaling(
            susc=self.susc,
            base_r0=self.base_r0,
            base_frac=self.base_frac,
            base_on=self.base_on
        )

        loader = self.CoordinateLoader()

        for country in scaling.country_names:
            coord = loader.get_coordinates(country)
            if coord:
                print(f"{country}: Latitude = {coord[0]}, Longitude = {coord[1]}")
            else:
                print(f"{country}: Coordinate not found!")

        scaling.prepare_clustering_data()

        cluster = self.Hierarchical(
            c_mtx_gen=scaling,
            country_names=scaling.country_names,
            img_prefix="contact_matrices",
            dist="euclidean"
        )

        plotter = self.Plotter(
            scaling_object=scaling,
            countries_to_plot=self.countries_to_plot,
            base_frac=self.base_frac,
            base_on="r0",
            base_r0=self.base_r0,
            coordinate_loader=loader,
            save_dir="figures"
        )

        plotter.run()
        plotter.plot_selected_country_contact_matrices(data=data)
        plotter.plot_distance_matrix(cluster)

        columns, dt, res = cluster.calculate_ordered_distance_matrix(
            threshold=self.threshold)
        plotter.plot_ordered_distance_matrix(dt, columns)

        leaf_colors = plotter.plot_dendrogram_with_threshold(
            res=res,
            threshold=self.threshold,
            labels=columns
        )

        plotter.plot_africa_map_with_epidemic_indicators(leaf_colors)
        plotter.plot_deaths_vs_beta_scatter(leaf_colors)
