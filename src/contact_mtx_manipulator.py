import numpy as np
from src.apvd import APVDReduction
from src.dataloader import DataLoader
from src.model.simulation import Simulation


class ContactMatrixScaling:
    def __init__(self, susc: float = 1.0, base_r0: float = 3.68,
                 base_frac: float = 0.5, base_on: str = "r0"):
        self.data = DataLoader()
        self.country_names = list(self.data.age_data.keys())

        self.setting_contacts = {}
        self.susc = susc
        self.base_on = base_on
        self.base_r0 = base_r0
        self.base_frac = base_frac

        self.scaled_full_matrices = []
        self.data_clustering = None

        self.run()

        self.population = {
            country: self.setting_contacts[country]["age_vector"].sum()
            for country in self.country_names
        }

        self.apply_apvd()

    def run(self):
        susceptibility = np.ones(16)
        susceptibility[:4] = self.susc

        for country in self.country_names:
            # Load contact matrices
            h_contact = self.data.contact_data[country]["home"]
            s_contact = self.data.contact_data[country]["school"]
            w_contact = self.data.contact_data[country]["work"]
            o_contact = self.data.contact_data[country]["other"]
            all_contact = h_contact + s_contact + w_contact + o_contact

            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))

            simulation = Simulation(
                data=self.data,
                contact_matrix=all_contact,
                age_vector=age_vector,
                susceptibility=susceptibility,
                base_r0=self.base_r0,
                base_frac=self.base_frac,
                base_on=self.base_on,
                country=country
            )

            beta = simulation.beta

            scaled_full = beta * all_contact
            scaled_home = beta * h_contact
            scaled_school = beta * s_contact
            scaled_work = beta * w_contact
            scaled_other = beta * o_contact

            self.scaled_full_matrices.append(scaled_full)
            self.setting_contacts[country] = {
                "beta": beta,
                "age_vector": age_vector,
                "contact_full": scaled_full,
                "contact_home": scaled_home,
                "contact_school": scaled_school,
                "contact_work": scaled_work,
                "contact_other": scaled_other,
                "contact_full_unscaled": all_contact
            }

    def apply_apvd(self):
        """Apply APVD reduction to full scaled matrices."""
        scaled_full_array = np.stack(self.scaled_full_matrices)  # (N, 16, 16)
        apvd = APVDReduction(contact_matrices=scaled_full_array, k_u=2, k_v=2)
        reduced_matrices = apvd.run()  # (N, 2, 2)

        for idx, country in enumerate(self.country_names):
            self.setting_contacts[country]["contact_full_reduced"] = reduced_matrices[idx]

    def prepare_clustering_data(self):
        """Flatten reduced matrices for clustering (Nx4 matrix)."""
        flattened = []
        for country in self.country_names:
            reduced = self.setting_contacts[country]["contact_full_reduced"]
            flattened.append(reduced.flatten())
        self.data_clustering = np.vstack(flattened)



