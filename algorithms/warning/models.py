import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 2. 宽度学习系统
class BroadLearningSystem:
    def __init__(self, n_feature_nodes=20, n_enhancement_nodes=50,
                 activation='sigmoid', reg_param=0.01):
        self.n_feature_nodes = n_feature_nodes
        self.n_enhancement_nodes = n_enhancement_nodes
        self.activation = activation
        self.reg_param = reg_param
        self.W_f = None
        self.W_e = None
        self.W_out = None
        self.b_f = None
        self.feature_nodes = []
        self.enhancement_nodes = []
        self.node_importance = {}
        self.input_dim = None

    def _activation_func(self, X: np.ndarray, func_name: str) -> np.ndarray:
        if func_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(X, -100, 100)))
        elif func_name == 'relu':
            return np.maximum(0, X)
        elif func_name == 'tanh':
            return np.tanh(X)
        else:
            return X

    def feature_mapping(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([])
        if self.W_f is None:
            n_features = X.shape[1]
            self.input_dim = n_features
            self.W_f = np.random.randn(n_features, self.n_feature_nodes) * 0.1
            self.b_f = np.random.randn(self.n_feature_nodes) * 0.1
        Z = np.dot(X, self.W_f) + self.b_f
        Z = self._activation_func(Z, self.activation)
        if len(self.feature_nodes) < 100:
            self.feature_nodes.append(Z)
        return Z

    def enhancement_mapping(self, Z: np.ndarray) -> np.ndarray:
        if Z.size == 0:
            return np.array([])
        if self.W_e is None:
            self.W_e = np.random.randn(self.n_feature_nodes, self.n_enhancement_nodes) * 0.1
        H = np.dot(Z, self.W_e)
        H = self._activation_func(H, 'relu')
        if len(self.enhancement_nodes) < 100:
            self.enhancement_nodes.append(H)
        return H

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0 or y.size == 0:
            return self
        Z = self.feature_mapping(X)
        H = self.enhancement_mapping(Z)
        A = np.hstack([Z, H])
        ATA = np.dot(A.T, A)
        I = np.eye(ATA.shape[0])
        ATA_reg = ATA + self.reg_param * I
        ATY = np.dot(A.T, y)
        self.W_out = np.linalg.solve(ATA_reg, ATY)
        self._compute_node_importance(A, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.array([])
        Z = self.feature_mapping(X)
        if Z.size == 0:
            return np.array([])
        H = self.enhancement_mapping(Z)
        if H.size == 0:
            return np.array([])
        A = np.hstack([Z, H])
        return np.dot(A, self.W_out)

    def incremental_learning(self, X_new: np.ndarray, y_new: np.ndarray,
                             mode='weight_update', n_new_nodes=10):
        if X_new.size == 0 or y_new.size == 0:
            return self
        if mode == 'weight_update':
            self._rls_update(X_new, y_new)
        elif mode == 'structure_expansion':
            self._add_enhancement_nodes(X_new, y_new, n_new_nodes)
        return self

    def _rls_update(self, X_new: np.ndarray, y_new: np.ndarray):
        Z_new = self.feature_mapping(X_new)
        H_new = self.enhancement_mapping(Z_new)
        A_new = np.hstack([Z_new, H_new])
        n_features = A_new.shape[1]
        if self.W_out is None:
            self.W_out = np.zeros((n_features, 1 if y_new.ndim == 1 else y_new.shape[1]))
        P = np.eye(n_features) * 1000
        for i in range(len(A_new)):
            a = A_new[i:i+1].T
            if y_new.ndim == 1:
                error = y_new[i] - np.dot(a.T, self.W_out).flatten()
            else:
                error = y_new[i] - np.dot(a.T, self.W_out)
            denominator = 1 + np.dot(a.T, np.dot(P, a))
            if denominator > 1e-10:
                K = np.dot(P, a) / denominator
                self.W_out += np.dot(K, error.reshape(1, -1))
                P = P - np.dot(K, np.dot(a.T, P))

    def _add_enhancement_nodes(self, X_new: np.ndarray, y_new: np.ndarray, n_new_nodes: int):
        Z_new = self.feature_mapping(X_new)
        W_e_new = np.random.randn(self.n_feature_nodes, n_new_nodes) * 0.1
        H_new = np.dot(Z_new, W_e_new)
        H_new = self._activation_func(H_new, 'relu')
        if self.W_e is not None:
            self.W_e = np.hstack([self.W_e, W_e_new])
        else:
            self.W_e = W_e_new
        self.n_enhancement_nodes += n_new_nodes
        H_all = np.dot(Z_new, self.W_e)
        H_all = self._activation_func(H_all, 'relu')
        A = np.hstack([Z_new, H_all])
        ATA = np.dot(A.T, A)
        I = np.eye(ATA.shape[0])
        ATA_reg = ATA + self.reg_param * I
        if y_new.ndim == 1:
            ATY = np.dot(A.T, y_new)
            self.W_out = np.linalg.solve(ATA_reg, ATY)
        else:
            self.W_out = np.linalg.solve(ATA_reg, np.dot(A.T, y_new))

    def _compute_node_importance(self, A: np.ndarray, y: np.ndarray):
        if self.W_out is None or A.size == 0:
            return
        importance = np.abs(self.W_out).sum(axis=1)
        total_importance = importance.sum()
        if total_importance > 0:
            for i, imp in enumerate(importance):
                self.node_importance[f'node_{i}'] = imp / total_importance

    def prune_nodes(self, threshold=0.05):
        if not self.node_importance:
            return len(self.W_out) if self.W_out is not None else 0
        important_indices = [i for i, (_, imp) in enumerate(self.node_importance.items()) if imp >= threshold]
        if self.W_out is not None and important_indices:
            self.W_out = self.W_out[important_indices, :]
            n_important = len(important_indices)
            self.n_enhancement_nodes = max(0, n_important - self.n_feature_nodes)
            return n_important
        return 0