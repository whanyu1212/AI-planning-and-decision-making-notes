import numpy as np


def compute_value_mrp(P, R, gamma, theta=1e-6):
    """
    Computes the state-value function V for an MRP using iterative dynamic programming.

    Parameters:
    - P: Transition probability matrix (n x n numpy array)
    - R: Reward vector (n-dimensional numpy array)
    - gamma: Discount factor (float)
    - theta: Convergence threshold (float)

    Returns:
    - V: Value function vector (n-dimensional numpy array)
    """
    n = len(R)
    V = np.zeros(n)

    while True:
        delta = 0.0
        V_new = np.copy(V)

        for s in range(n):
            # Bellman update: V(s) = R(s) + gamma * sum(P[s, s'] * V(s'))
            V_new[s] = R[s] + gamma * np.dot(P[s], V)
            delta = max(delta, abs(V[s] - V_new[s]))

        V = V_new
        # The algorithm has converged enough
        if delta < theta:
            break
    return V


# Example usage:
if __name__ == "__main__":
    # Example 1:
    # P(S1|S1) = 0.5, P(S2|S1) = 0.5
    # P(S1|S2) = 0.2, P(S2|S2) = 0.8
    # R(S1) = 1, R(S2) = 2
    # gamma = 0.9
    P = np.array([[0.5, 0.5], [0.2, 0.8]])
    R = np.array([1, 2])
    gamma = 0.9

    V = compute_value_mrp(P, R, gamma)
    print("Value Function:", V)
