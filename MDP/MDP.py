import numpy as np


def compute_value_mdp(P, R, gamma, theta=1e-6):
    """
    Performs value iteration for an MDP.

    Parameters:
    - P: Transition probability array of shape (n, m, n), where:
         P[s, a, s'] is the probability of transitioning from state s to state s' given action a.
    - R: Reward array of shape (n, m, n), where:
         R[s, a, s'] is the reward received when transitioning from state s to state s' given action a.
    - gamma: Discount factor.
    - theta: Convergence threshold.

    Returns:
    - V: Optimal value function vector (n-dimensional).
    - policy: Optimal policy vector (n-dimensional) with action indices.
    """
    n, m, _ = P.shape
    V = np.zeros(n)
    policy = np.zeros(n, dtype=int)

    while True:
        delta = 0
        for s in range(n):
            # Compute Q(s,a) for all actions a in state s.
            Q_sa = np.zeros(m)
            for a in range(m):
                # sum of P[s, a, s'] * (R[s, a, s'] + gamma * V[s'])
                # which is the expected value of taking action a in state s.
                Q_sa[a] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
            # we want to find the maximum Q(s,a) for all actions a in state s.
            max_val = np.max(Q_sa)
            delta = max(delta, abs(max_val - V[s]))
            # state value function V(s) is the maximum Q(s,a) for all actions a in state s.
            # the optimal action is the action that maximizes Q(s,a) at state s.
            V[s] = max_val
            policy[s] = np.argmax(
                Q_sa
            )  # returns the index of the maximum value in Q_sa.
        if delta < theta:
            break

    return V, policy


# Example usage:
if __name__ == "__main__":
    # Define a simple MDP with 3 states and 2 actions.
    P = np.array(
        [
            # Transitions for state 0:
            [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]],  # Action 0  # Action 1
            # Transitions for state 1:
            [[0.0, 0.6, 0.4], [0.0, 0.3, 0.7]],  # Action 0  # Action 1
            # Transitions for state 2 (terminal state)
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],  # Action 0  # Action 1
        ]
    )

    R = np.array(
        [
            # Rewards for state 0:
            [[5, 10, 0], [3, 8, 0]],  # Action 0  # Action 1
            # Rewards for state 1:
            [[0, 2, 4], [0, 1, 7]],  # Action 0  # Action 1
            # Rewards for state 2 (terminal state)
            [[0, 0, 0], [0, 0, 0]],  # Action 0  # Action 1
        ]
    )

    gamma = 0.9

    V, policy = compute_value_mdp(P, R, gamma)
    print("Optimal Value Function:", V)
    print("Optimal Policy:", policy)
