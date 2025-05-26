import numpy as np
from src.model.epid_model import EpidemicModel
from src.model.r0_generator import R0Model


class Simulation:
    def __init__(self, data, contact_matrix, age_vector, susceptibility,
                 base_r0: float = 3.68, base_frac: float = 0.5,
                 base_on: str = "r0", country="Unknown"):

        self.data = data
        self.contact_matrix = contact_matrix
        self.age_vector = age_vector
        self.country = country
        self.base_on = base_on
        self.base_r0 = base_r0
        self.base_frac = base_frac
        self.n_age = 16

        self.model = EpidemicModel(model_data=data, country=country)
        self.population = self.model.population
        self.pop_total = self.population.sum()

        self.params = self.data.model_parameters_data.copy()
        self.params.update({"susc": susceptibility})

        self.init_values = self.model.get_initial_values()
        self.t = np.arange(0, 1000, 1.0)

        self.susceptibles = self.init_values[
            self.model.c_idx["s"] * self.n_age: (self.model.c_idx["s"] + 1) * self.n_age
        ] / self.population

        self.beta = self.determine_beta()
        self.params.update({"beta_a": self.beta, "beta_m": self.beta})

    def determine_beta(self):
        if self.base_on == "r0":
            r0gen = R0Model(param=self.params, n_age=self.n_age)
            r0_current = r0gen.get_eig_val(
                susceptibles=self.susceptibles.reshape((1, -1)),
                population=self.population,
                contact_mtx=self.contact_matrix
            )
            return self.base_r0 / r0_current[0]

        else:
            print(f"🔍 Searching for β to achieve {self.base_frac * 100:.2f}% "
                  f"{self.base_on} in {self.country}")

            # Estimate initial beta based on R0
            r0gen = R0Model(param=self.params, n_age=self.n_age)
            r0_current = r0gen.get_eig_val(
                susceptibles=self.susceptibles.reshape((1, -1)),
                population=self.population,
                contact_mtx=self.contact_matrix
            )[0]
            beta_r0 = self.base_r0 / r0_current

            beta_min = max(0.0001, beta_r0 * 0.2)
            beta_max = beta_r0 * 2.5
            best_beta = None

            for beta in np.linspace(beta_min, beta_max, 80):
                self.params.update({"beta_a": beta, "beta_m": beta})
                sol = self.model.get_solution(
                    init_values=self.init_values,
                    t=self.t,
                    parameters=self.params,
                    cm=self.contact_matrix
                )

                if self.base_on == "infected":
                    output = self.model.get_infected(sol).max()
                elif self.base_on == "peak":
                    output = self.model.get_epidemic_peak(sol)
                elif self.base_on == "hospital":
                    output = self.model.get_hospitalized(sol).max()
                elif self.base_on == "icu":
                    output = self.model.get_icu(sol).max()
                elif self.base_on == "deaths":
                    output = self.model.get_deaths(sol)[-1]
                else:
                    raise ValueError(f"Unknown base_on option: {self.base_on}")

                frac = output / (self.pop_total + 1e-9)
                print(f"  β = {beta:.3f} → {frac:.3%} of population")

                if frac >= self.base_frac:
                    best_beta = beta
                    print(f"✅ Reached target at β = {beta:.3f} ({frac:.2%})\n")
                    break

            if best_beta is None:
                raise RuntimeError(
                    f"❌ No β found that reaches {self.base_frac * 100:.1f}% "
                    f"{self.base_on} for {self.country}"
                )

            return best_beta

    def run_simulation(self):
        sol = self.model.get_solution(
            init_values=self.init_values,
            t=self.t,
            parameters=self.params,
            cm=self.contact_matrix
        )
        return sol

    def get_summary_outputs(self, sol):
        return {
            "infected": self.model.get_infected(sol),
            "hospitalized": self.model.get_hospitalized(sol),
            "icu": self.model.get_icu(sol),
            "deaths": self.model.get_deaths(sol),
        }
