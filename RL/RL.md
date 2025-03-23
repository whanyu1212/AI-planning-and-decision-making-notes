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
