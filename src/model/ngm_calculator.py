import torch
from scipy.linalg import block_diag


class NGMCalculator:
    def __init__(self, param: dict, n_age: int) -> None:
        self.ngm_small_mtx = None
        states = ["e", "d", "a"]
        self.n_states = len(states)
        self.n_age = n_age
        self.parameters = param

        self.i = {states[index]: index for index in range(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self.symmetric_contact_matrix = None

    def _idx(self, state: str) -> torch.Tensor:
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        f = torch.zeros((self.n_age * n_states, self.n_age * n_states))

        f[i["e"]:s_mtx:n_states, i["d"]:s_mtx:n_states] = self.parameters["epsilon_d"] * \
                                                          contact_mtx.T
        f[i["e"]:s_mtx:n_states, i["a"]:s_mtx:n_states] = self.parameters["epsilon_a"] * \
                                                          contact_mtx.T
        return f

    def _get_v(self):
        idx = self._idx
        v = torch.zeros((self.n_age * self.n_states, self.n_age * self.n_states))

        # E -> E (exposed to exposed)
        v[idx("e"), idx("e")] = self.parameters["sigma"]

        # E -> A (exposed to asymptomatic)
        v[idx("a"), idx("e")] = -(1 - self.parameters["delta"]) * self.parameters["sigma"]

        # E -> D (exposed to symptomatic)
        v[idx("d"), idx("e")] = -self.parameters["delta"] * self.parameters["sigma"]

        # A -> A (asymptomatic to asymptomatic)
        v[idx("a"), idx("a")] = self.parameters["gamma"]

        # D -> D (symptomatic to symptomatic)
        v[idx("d"), idx("d")] = self.parameters["gamma"]

        self.v_inv = torch.linalg.inv(v)

    def _get_e(self):
        block = torch.zeros(self.n_states)
        block[0] = 1
        self.e = block
        for _ in range(1, self.n_age):
            self.e = block_diag(self.e, block)
            self.e = torch.tensor(self.e, dtype=torch.float32)

    def run(self, symmetric_contact_mtx: torch.Tensor):
        f = self._get_f(contact_mtx=symmetric_contact_mtx)
        ngm_large = torch.matmul(f, self.v_inv)
        self.ngm_small_mtx = torch.matmul(
            torch.matmul(self.e, ngm_large),
            self.e.T
        )

