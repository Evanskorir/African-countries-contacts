import numpy as np
from src.model.model_base import EpidemicModelBase


class EpidemicModel(EpidemicModelBase):
    def __init__(self, model_data, country) -> None:
        compartments = ["s", "e", "a", "m", "h", "c", "r"]
        super().__init__(model_data=model_data, compartments=compartments, country=country)

        self.params = model_data.model_parameters_data  # store model parameters like delta_c

    def update_initial_values(self, iv: dict):
        iv["e"][1] = 1  # Seed infection
        iv.update({"s": self.population - (iv["e"] + iv["a"] + iv["m"] +
                                           iv["h"] + iv["c"] + iv["r"])})

    def get_model(self, xs: np.ndarray, t, ps: dict, cm: np.ndarray) -> np.ndarray:
        s, e, a, m, h, c, r = xs.reshape(-1, self.n_age)

        # Transmission terms
        transmission = ps["beta_a"] * np.array(a).dot(cm) + ps["beta_m"] * \
                       np.array(m).dot(cm)

        model_eq_dict = {
            "s": -transmission * s / self.population,
            "e": transmission * s / self.population - ps["sigma"] * e,
            "a": ps["q"] * ps["sigma"] * e - ps["gamma_a"] * a,
            "m": (1 - ps["q"]) * ps["sigma"] * e - (ps["kappa"] + ps["gamma_m"]) * m,
            "h": ps["kappa"] * m + ps["phi"] * c - (ps["xi"] + ps["gamma_h"]) * h,
            "c": ps["xi"] * h - (ps["delta_c"] + ps["phi"]) * c,
            "r": ps["gamma_a"] * a + ps["gamma_m"] * m + ps["gamma_h"] * h,
        }
        return self.get_array_from_dict(comp_dict=model_eq_dict)

    def get_infected(self, solution: np.ndarray) -> np.ndarray:
        idx_e = self.c_idx["e"]
        idx_a = self.c_idx["a"]
        idx_m = self.c_idx["m"]
        return (self.aggregate_by_age(solution, idx_e) +
                self.aggregate_by_age(solution, idx_a) +
                self.aggregate_by_age(solution, idx_m))

    def get_epidemic_peak(self, solution: np.ndarray) -> float:
        return self.get_infected(solution).max()

    def get_hospitalized(self, solution: np.ndarray) -> np.ndarray:
        idx_h = self.c_idx["h"]
        return self.aggregate_by_age(solution, idx_h)

    def get_icu(self, solution: np.ndarray) -> np.ndarray:
        idx_c = self.c_idx["c"]
        return self.aggregate_by_age(solution, idx_c)

    def get_deaths(self, solution: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Calculate cumulative deaths from ICU over time."""
        idx_c = self.c_idx["c"]
        c_values = self.aggregate_by_age(solution, idx_c)  # ICU cases over time

        # delta_c is vector (age-dependent mortality)
        delta_c = self.params["delta_c"]
        if isinstance(delta_c, np.ndarray) or isinstance(delta_c, list):
            delta_c = np.array(delta_c).mean()

        deaths = np.cumsum(delta_c * c_values * dt)
        return deaths
