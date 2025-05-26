import numpy as np
from abc import ABC, abstractmethod


class R0GeneratorBase(ABC):
    def __init__(self, param: dict, states: list, n_age: int) -> None:
        self.states = states
        self.n_age = n_age
        self.parameters = param
        self.n_states = len(self.states)
        self.i = {state: idx for idx, state in enumerate(states)}
        self.s_mtx = n_age * self.n_states

        self.v_inv = None
        self.e = None

    def _idx(self, state: str) -> np.ndarray:
        return np.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(self, susceptibles: np.ndarray, population: np.ndarray, contact_mtx: np.ndarray):
        """Calculate the largest eigenvalue of the next-generation matrix (NGM)."""
        # No scaling of contact matrix here anymore!
        cm_tensor = np.tile(contact_mtx, (susceptibles.shape[0], 1, 1))
        susc_tensor = susceptibles.reshape((susceptibles.shape[0], susceptibles.shape[1], 1))
        contact_matrix_tensor = cm_tensor * susc_tensor

        eig_val_eff = []
        for cm in contact_matrix_tensor:
            f = self._get_f(cm)
            ngm_large = f @ self.v_inv
            ngm = self.e @ ngm_large @ self.e.T
            eig_val = np.sort(np.abs(np.linalg.eigvals(ngm)))
            eig_val_eff.append(float(eig_val[-1]))

        return eig_val_eff

    @abstractmethod
    def _get_e(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_v(self):
        pass

    @abstractmethod
    def _get_f(self, contact_matrix: np.ndarray):
        pass

