import numpy as np


class APVDReduction:
    def __init__(self, contact_matrices: np.ndarray, k_u: int = 2, k_v: int = 2):
        """
        contact_matrices: (N, 16, 16) numpy array where N = number of countries
        """
        self.contact_matrices = contact_matrices
        self.k_u = k_u
        self.k_v = k_v

        self.L = None
        self.R = None
        self.W_list = []

    def first_stage_svd(self):
        U_list = []
        V_list = []

        for matrix in self.contact_matrices:
            U, D, Vt = np.linalg.svd(matrix, full_matrices=False)

            # Correct: scale by multiplying with diagonal matrix of singular values
            U_scaled = U[:, :self.k_u] @ np.diag(D[:self.k_u])
            V_scaled = Vt.T[:, :self.k_v] @ np.diag(D[:self.k_v])

            U_list.append(U_scaled)
            V_list.append(V_scaled)

        return np.hstack(U_list), np.hstack(V_list)

    def second_stage_group_svd(self, P, Q):
        Up, _, _ = np.linalg.svd(P, full_matrices=False)
        Uq, _, _ = np.linalg.svd(Q, full_matrices=False)

        self.L = Up[:, :2]
        self.R = Uq[:, :2]

    def project(self):
        for matrix in self.contact_matrices:
            W = self.L.T @ matrix @ self.R
            self.W_list.append(W)

    def run(self):
        P, Q = self.first_stage_svd()
        self.second_stage_group_svd(P, Q)
        self.project()
        return np.array(self.W_list)  # shape: (N, 2, 2)

