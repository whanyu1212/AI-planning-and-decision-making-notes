<!-- omit in toc -->
# Reinforcement Learning

Cn view it as a form of Policy evaluation without known dynamics & reward models for simplicity

<!-- omit in toc -->
## Table of Contents
- [Key Points](#key-points)
- [Monte Carlo (MC) Policy Evaluation](#monte-carlo-mc-policy-evaluation)
  - [First Visit MC](#first-visit-mc)
  - [Every Visit MC](#every-visit-mc)
  - [Example 1: First-Visit Monte Carlo](#example-1-first-visit-monte-carlo)
  - [Example 2: Every-Visit Monte Carlo](#example-2-every-visit-monte-carlo)
  - [On Policy Monte Carlo](#on-policy-monte-carlo)
  - [On Policy vs Off Policy](#on-policy-vs-off-policy)
- [Temporal-Difference (TD) Learning](#temporal-difference-td-learning)
  - [Tabular TD(0) for Estimating (V^\\pi)](#tabular-td0-for-estimating-vpi)
- [SARSA: On-Policy TD Control](#sarsa-on-policy-td-control)
- [Q-Learning: Off Policy TD Control](#q-learning-off-policy-td-control)


---

### Key Points

**Policy Evaluation:**
- In classical policy evaluation (with known dynamics), you compute the value function for a given policy using the model.
- In RL, techniques like temporal-difference (TD) learning allow you to estimate the value function by interacting with the environment, even when the transition probabilities \(P(s'|s,a)\) and reward function \(R(s,a)\) are unknown.

**Beyond Policy Evaluation:**
- RL doesn't stop at evaluating a fixed policy.
- It also seeks to improve the policy based on the estimated value function.
- Methods such as Q-learning, SARSA, and actor-critic algorithms iteratively evaluate and improve the policy to converge to optimal behavior.

**Model-Free vs. Model-Based:**
- The statement more closely describes model-free RL.
- In model-based RL, even though the dynamics might initially be unknown, the agent tries to learn a model of the environment and then uses it for planning and policy evaluation.

---

### Monte Carlo (MC) Policy Evaluation
- \( V^\pi(s) = \mathbb{E}_{\tau \sim \pi}[G_t \mid s_t = s] \)
- Simple idea: Value = Mean return
- If trajectories are all finite, sample set of trajectories & average returns
- All trajectories may not be same length
- Does not require MDP dynamics/rewards
- Does not assume state is Markov
- Can be applied to episodic MDPs
  - Averaging over returns from a complete episode
  - Requires each episode to terminate

#### First Visit MC
1. **Initialize**  
   \[
   N(s) = 0, \quad G(s) = 0 \quad \forall s \in S
   \]
   
   Where:
   - **\( N(s) \)**: A counter that tracks how many times state \( s \) has been *first-visited* across all sampled episodes.  
     - In *first-visit* Monte Carlo methods, you only update a state's value the first time you encounter it within an episode.
   - **\( G(s) \)**: The cumulative sum of returns from all *first visits* to state \( s \).  
     - Every time you encounter state \( s \) for the first time in an episode, you add the *return* from that time step to \( G(s) \).

2. **Loop**  
   1. **Sample episode \( i \)**  
      \[
      s_{i,1},\, a_{i,1},\, r_{i,1},\, s_{i,2},\, a_{i,2},\, r_{i,2},\, \ldots,\, s_{i,T_i},\, a_{i,T_i},\, r_{i,T_i}
      \]

   2. **Define**  
      \[
      G_{i,t} = r_{i,t} + \gamma\,r_{i,t+1} + \gamma^2\,r_{i,t+2} + \ldots + \gamma^{T_i - t}\,r_{i,T_i}
      \]  
      as the return from time step \( t \) onward in the \( i \)-th episode.

   3. **For each time step** \( t \) **until** \( T_i \) (the end of the \( i \)-th episode):
      - **If this is the first time that** \( s_{i,t} \) **is visited in the \( i \)-th episode**:
        1. Increment the counter of total first visits:  
           \[
           N(s_{i,t}) = N(s_{i,t}) + 1
           \]
        2. Increment the total return:  
           \[
           G(s_{i,t}) = G(s_{i,t}) + G_{i,t}
           \]
        3. Update the estimate:  
           \[
           V^*(s_{i,t}) = \frac{G(s_{i,t})}{N(s_{i,t})}
           \]

#### Every Visit MC

1. **Initialize**  
   \[
   N(s) = 0, \quad G(s) = 0 \quad \forall s \in S
   \]

2. **Loop**  
   1. **Sample episode** \(i\)  
      \[
      s_{i,1},\, a_{i,1},\, r_{i,1},\, s_{i,2},\, a_{i,2},\, r_{i,2},\, \ldots,\, s_{i,T_i},\, a_{i,T_i},\, r_{i,T_i}
      \]

   2. **Define**  
      \[
      G_{i,t} = r_{i,t} + \gamma \, r_{i,t+1} + \gamma^2 \, r_{i,t+2} + \ldots + \gamma^{T_i - t} \, r_{i,T_i}
      \]  
      as the return from time step \(t\) onward in the \(i\)-th episode.

   3. **For each time step** \(t\) **until** \(T_i\) (the end of the \(i\)-th episode):
      - **If this is the first time that** \(s_{i,t}\) **is visited in the \(i\)-th episode**:
        1. Increment the counter of total first visits:  
           \[
           N\bigl(s_{i,t}\bigr) = N\bigl(s_{i,t}\bigr) + 1
           \]
        2. Increment the total return:  
           \[
           G\bigl(s_{i,t}\bigr) = G\bigl(s_{i,t}\bigr) + G_{i,t}
           \]
        3. Update the estimate:  
           \[
           V^*\bigl(s_{i,t}\bigr) = \frac{G\bigl(s_{i,t}\bigr)}{N\bigl(s_{i,t}\bigr)}
           \]




#### Example 1: First-Visit Monte Carlo

**Episode 1:**  
- **Sequence:** \(A \to B \to A \to T\)  
- **Rewards:** \(r_1 = 2,\; r_2 = 3,\; r_3 = 1\)  
- **First-Visit Returns:**  
  - For state \(A\) (first seen at \(t=1\)):  
    \[
    G_A^{(1)} = 2 + 3 + 1 = 6
    \]
  - For state \(B\) (first seen at \(t=2\)):  
    \[
    G_B^{(1)} = 3 + 1 = 4
    \]

**Episode 2:**  
- **Sequence:** \(B \to A \to T\)  
- **Rewards:** \(r_1 = 1,\; r_2 = 4\)  
- **First-Visit Returns:**  
  - For state \(B\) (first seen at \(t=1\)):  
    \[
    G_B^{(2)} = 1 + 4 = 5
    \]
  - For state \(A\) (first seen at \(t=2\)):  
    \[
    G_A^{(2)} = 4
    \]

**Value Estimates:**  
- For \(A\):
  \[
  V(A) = \frac{G_A^{(1)} + G_A^{(2)}}{2} = \frac{6 + 4}{2} = 5
  \]
- For \(B\):
  \[
  V(B) = \frac{G_B^{(1)} + G_B^{(2)}}{2} = \frac{4 + 5}{2} = 4.5
  \]

---

#### Example 2: Every-Visit Monte Carlo

**Episode 1:**  
- **Sequence:** \(A \to B \to A \to T\)  
- **Rewards:** \(r_1 = 2,\; r_2 = 3,\; r_3 = 1\)  
- **Every-Visit Returns:**  
  - For state \(A\):  
    - At first occurrence (\(t=1\)):
      \[
      G_{A,1}^{(1)} = 2 + 3 + 1 = 6
      \]
    - At second occurrence (\(t=3\)):
      \[
      G_{A,2}^{(1)} = 1
      \]
  - For state \(B\) (only occurrence at \(t=2\)):  
    \[
    G_B^{(1)} = 3 + 1 = 4
    \]

**Episode 2:**  
- **Sequence:** \(B \to A \to T\)  
- **Rewards:** \(r_1 = 1,\; r_2 = 4\)  
- **Every-Visit Returns:**  
  - For state \(B\) (only occurrence at \(t=1\)):
    \[
    G_B^{(2)} = 1 + 4 = 5
    \]
  - For state \(A\) (only occurrence at \(t=2\)):
    \[
    G_A^{(2)} = 4
    \]

**Value Estimates:**  
- For \(A\) (all returns: \(6\), \(1\), and \(4\)):
  \[
  V(A) = \frac{6 + 1 + 4}{3} = \frac{11}{3} \approx 3.67
  \]
- For \(B\) (all returns: \(4\) and \(5\)):
  \[
  V(B) = \frac{4 + 5}{2} = 4.5
  \]


#### On Policy Monte Carlo
a method for finding an optimal policy in a Markov Decision Process (MDP) by repeatedly:

- **Sampling episodes** under the current policy (the same policy that will be improved).
- **Estimating action values** (\(Q^\pi(s, a)\)) from returns observed in those episodes.
- **Improving the policy** to become greedy (or \(\varepsilon\)-greedy) with respect to those action-value estimates.

1. Initialization

- Start with an arbitrary policy \(\pi\) that explores all actions (e.g., \(\varepsilon\)-soft).
- Initialize the action-value function \(Q(s,a)\) arbitrarily.

2. Generate Episodes

- Use \(\pi\) to interact with the environment for one episode.
- Record each visited state-action pair \((s,a)\) and the return that follows it.

3. Action-Value Estimation

- For each \((s,a)\) visited in the episode, update the estimate of \(Q(s,a)\) by averaging all the returns that started from that state-action pair.
  
  - **Incremental Update Form:**
    \[
    Q(s,a) \; \leftarrow \; Q(s,a) + \alpha \Bigl(\text{Return} - Q(s,a)\Bigr)
    \]
  
  - **Monte Carlo Average:**
    \[
    Q(s,a) = \frac{\sum \text{(all returns from }(s,a)\text{)}}{\text{number of times }(s,a)\text{ has been visited}}
    \]
  
- “First-visit” MC updates use only the first time \((s,a)\) appears in an episode; “every-visit” MC updates for every occurrence of \((s,a)\).

4. Policy Improvement

- For each state \(s\), improve \(\pi\) to be greedy or \(\varepsilon\)-greedy with respect to the updated \(Q\):
  \[
  \pi(s) =
    \begin{cases}
      \arg \max_a\, Q(s,a) & \text{with probability } 1-\varepsilon, \\
      \text{random action} & \text{with probability } \varepsilon.
    \end{cases}
  \]
- This ensures that all actions continue to be explored (so you don’t prematurely converge to a suboptimal policy).

5. Repeat

- Continue generating episodes using the updated policy, estimating \(Q\), and improving \(\pi\).
- Over many iterations, \(\pi\) converges to an optimal policy (assuming sufficient exploration).

#### On Policy vs Off Policy
- **Target Policy (π)**: The policy being evaluated and improved
- **Behavior Policy (b)**: The policy that generates trajectories (the past experience)
- **On Policy**: When π = b (the same policy is used for both decision-making and learning)
- **Off Policy**: When π ≠ b (learning about one policy while following another)

<br>

*Why do we care about off policy RL?* 
- We want to use old experience to still improve the target policy
- We want b to be more exploratory than π (which out to be greedy)

---

### Temporal-Difference (TD) Learning

**TD learning** bridges the gap between **Monte Carlo** methods and **Dynamic Programming** by updating value estimates based on *bootstrapping*—using the current value estimates rather than waiting for the final return of an episode.

1. **Monte Carlo Update**

\[
V(s_t) \;\leftarrow\; V(s_t) \;+\; \alpha \,\bigl[G_t - V(s_t)\bigr]
\]

- \(G_t\) is the *full return* from time \(t\) to the end of the episode.  
- You only update \(V(s_t)\) after the entire episode completes (i.e., once \(G_t\) is known).

2. **Temporal-Difference (TD) Update**

\[
V(s_t) \;\leftarrow\; V(s_t) \;+\; \alpha \,\bigl[r_t + \gamma \, V(s_{t+1}) - V(s_t)\bigr]
\]

- You use the **immediate reward** \(r_t\) plus a **discounted estimate** \(\gamma \, V(s_{t+1})\) of the next state’s value.  
- The difference \(\bigl[r_t + \gamma \, V(s_{t+1}) - V(s_t)\bigr]\) is called the **TD error**.  
- You update **incrementally** at each time step, without waiting for the entire episode to finish.

<br>

**Key Takeaways**

- **Monte Carlo (MC) Methods**  
  - Wait until the episode ends to compute the true return \(G_t\).  
  - The update nudges \(V(s_t)\) toward that actual return.

- **TD Methods**  
  - Update after *every single step*, using the current value estimate \(V(s_{t+1})\) rather than the full return.  
  - The update nudges \(V(s_t)\) toward the bootstrapped target \(\bigl[r_t + \gamma \, V(s_{t+1})\bigr]\).

In many cases, **TD learning** can be more efficient because it updates more frequently and doesn’t require the episode to terminate before making a learning step.

#### Tabular TD(0) for Estimating \(V^\pi\)

**Input**: The policy \(\pi\) to be evaluated

1. **Initialize** \(V(s)\) arbitrarily (e.g., \(V(s) = 0\) for all \(s \in S^+\)).

2. **Repeat** (for each episode):
   1. **Initialize** \(S\).
   2. **Repeat** (for each step of the episode):
      - \(A \leftarrow\) action given by \(\pi\) for \(S\).
      - Take action \(A\); observe reward \(R\) and next state \(S'\).
      - \( V(S) \leftarrow V(S) + \alpha \bigl[R + \gamma \, V(S') - V(S)\bigr] \)
      - \(S \leftarrow S'\).
   3. **Until** \(S\) is terminal.

---

### SARSA: On-Policy TD Control
- An acronym for State–Action–Reward–State–Action
- At each step, it selects the next action using the same ε-greedy policy it’s learning about.
- Learns the value of the action actually taken by the current policy

After observing the transition \((S, A, R, S', A')\), the update for \(Q(S,A)\) is:

\( Q(S,A) \leftarrow Q(S,A) + \alpha \, [R + \gamma\,Q(S', A') - Q(S,A)] \).

Here, \(A'\) is chosen by the current \(\varepsilon\)-greedy policy from the next state \(S'\).


1. **Initialize** \(Q(s,a)\) for all \(s \in S, a \in A(s)\) arbitrarily, and set \(Q(\text{terminal-state}, \cdot) = 0\).

2. **Repeat** (for each episode):
   - **Initialize** \(S\).
   - **Choose** \(A\) from \(S\) using a policy derived from \(Q\) (e.g., \(\varepsilon\)-greedy).

   - **Repeat** (for each step of the episode):
     1. Take action \(A\), observe \(R\) and \(S'\).
     2. Choose \(A'\) from \(S'\) using a policy derived from \(Q\) (e.g., \(\varepsilon\)-greedy).
     3. Update:
        \( Q(S, A) \leftarrow Q(S, A) + \alpha \,\bigl[R + \gamma\,Q(S', A') - Q(S, A)\bigr] \)
     4. \(S \leftarrow S'\); \(A \leftarrow A'\).

   - **Until** \(S\) is terminal.

---

### Q-Learning: Off Policy TD Control
- **Action-Value Function \(Q(s,a)\):**  
  This function represents the expected return (sum of discounted rewards) when taking action \(a\) in state \(s\) and then following the optimal policy thereafter:
  \[
  Q^*(s,a) = \mathbb{E}\left[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a\right].
  \]

<br>

1. **Initialize** \(Q(s,a)\) for all \(s \in S, a \in A(s)\) arbitrarily, and set \(Q(\text{terminal-state}, \cdot) = 0\).

2. **Repeat** (for each episode):
   - **Initialize** \(S\).
   - **Choose** \(A\) from \(S\) using a policy derived from \(Q\) (e.g., \(\varepsilon\)-greedy).
   - **Take** action \(A\), observe \(R\) and \(S'\).
   - **Update**:
     \( Q(S, A) \leftarrow Q(S, A) + \alpha \,\bigl[R + \gamma \,\max_{a} Q(S', a) - Q(S, A)\bigr] \)
   - \(S \leftarrow S'\)

   - **Until** \(S\) is terminal.
