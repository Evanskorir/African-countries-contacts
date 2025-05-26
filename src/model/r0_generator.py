import numpy as np
from scipy.linalg import block_diag

from src.model.r0_generator_base import R0GeneratorBase


class R0Model(R0GeneratorBase):
    def __init__(self, param: dict, n_age: int = 16) -> None:
        states = ["e", "a", "m"]
        super().__init__(param=param, states=states, n_age=n_age)

        self._get_e()
        self._get_v()

    def _get_v(self) -> np.ndarray:
        idx = self._idx
        ps = self.parameters

        v = np.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (leaving due to incubation)
        v[idx("e"), idx("e")] = ps["sigma"]

        # E -> A (become asymptomatic infectious)
        v[idx("a"), idx("e")] = -ps["q"] * ps["sigma"]

        # E -> M (become mild symptomatic infectious)
        v[idx("m"), idx("e")] = -(1 - ps["q"]) * ps["sigma"]

        # A -> A (leaving due to recovery)
        v[idx("a"), idx("a")] = ps["gamma_a"]

        # M -> M (leaving due to progression/recovery)
        v[idx("m"), idx("m")] = ps["kappa"] + ps["gamma_m"]

        self.v_inv = np.linalg.inv(v)

    def _get_f(self, contact_mtx: np.ndarray) -> np.ndarray:
        idx = self._idx
        ps = self.parameters
        n_states = self.n_states

        susc_vec = np.ones(self.n_age)  # assuming susceptibility = 1 for all ages
        susc_vec = susc_vec.reshape((-1, 1))

        f = np.zeros((self.n_age * n_states, self.n_age * n_states))

        # Infections from asymptomatic A
        f[idx("e"), idx("a")] = (ps["beta_a"] * (contact_mtx.T @ susc_vec)).flatten()

        # Infections from mild symptomatic M
        f[idx("e"), idx("m")] = (ps["beta_m"] * (contact_mtx.T @ susc_vec)).flatten()

        return f

    def _get_e(self):
        block = np.zeros(self.n_states)
        block[0] = 1  # Only Exposed (E) are counted for new infections
        self.e = block
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)

