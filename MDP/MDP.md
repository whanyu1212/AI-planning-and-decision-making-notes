# Markov Decision Process

### <u>Definitions:</u>

**Markov Reward Processes (no actions & decisions)**

**Elements of MRP:**
$$M = (S, P, R, \gamma)$$

Where:
- **S**: Set of states
- **P**: State transition probability function 
  - $P(s'|s)$ = Probability of transitioning to state $s'$ from state $s$
- **R**: Reward function
  - $R(s)$ or $R(s,s')$ = Reward received when transitioning from state $s$ to $s'$
- **γ**: Discount factor
  - $0 \leq \gamma \leq 1$

**Horizon** (*H*)
- **Definition:** The number of time steps in each episode.
- **Notes:**
  - The horizon can be infinite.
  - If finite, the process is known as a *finite Markov Reward Process*.

**Return \( G_t \)**
- **Definition:** The discounted sum of rewards from time step \( t \) to the horizon \( H \).
- **Formula:**
  \[
  G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{H-1} r_{t+H-1}
  \]

**State Value Function \( V(s) \)**
- **Definition:** The expected return when starting in state \( s \).
- **Formula:**
  \[
  V(s) = \mathbb{E}[G_t \mid s_t = s] = \mathbb{E}\left[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{H-1} r_{t+H-1} \,\middle|\, s_t = s\right]
  \]

<br>

**State Value Function \(V(s)\)**

The state value function of *MRP* can be written as:

\[
V(s) = R(s) + \gamma \sum_{s' \in S} P(s' \mid s) V(s')
\]

Where:
- **\(R(s)\)** represents the *immediate reward*.
- **\(\gamma \sum_{s' \in S} P(s' \mid s) V(s')\)** represents the *discounted sum of future rewards*.

<br>

The Bellman equation in matrix form is:

\[
\mathbf{V} = \mathbf{R} + \gamma \mathbf{P}\mathbf{V}
\]

Where:
- **\(\mathbf{V}\)** is the column vector of state values:
  \[
  \mathbf{V} = \begin{bmatrix} V(s_1) \\ V(s_2) \\ \vdots \\ V(s_n) \end{bmatrix}
  \]
  Each \(V(s_i)\) is the expected return starting from state \(s_i\).

- **\(\mathbf{R}\)** is the column vector of immediate rewards:
  \[
  \mathbf{R} = \begin{bmatrix} R(s_1) \\ R(s_2) \\ \vdots \\ R(s_n) \end{bmatrix}
  \]
  Each \(R(s_i)\) is the immediate reward received in state \(s_i\).

- **\(\gamma\)** is the discount factor, which scales the importance of future rewards relative to immediate rewards.

- **\(\mathbf{P}\)** is the state transition probability matrix:
  \[
  \mathbf{P} = \begin{bmatrix}
  P(s_1 \mid s_1) & P(s_2 \mid s_1) & \cdots & P(s_n \mid s_1) \\
  P(s_1 \mid s_2) & P(s_2 \mid s_2) & \cdots & P(s_n \mid s_2) \\
  \vdots         & \vdots         & \ddots & \vdots         \\
  P(s_1 \mid s_n) & P(s_2 \mid s_n) & \cdots & P(s_n \mid s_n)
  \end{bmatrix}
  \]
  Each entry \(P(s_j \mid s_i)\) is the probability of transitioning from state \(s_i\) to state \(s_j\).

**Solving for \(\mathbf{V}\):**

Rearrange the equation:

\[
\mathbf{V} - \gamma \mathbf{P}\mathbf{V} = \mathbf{R}
\]

Factor out \(\mathbf{V}\):

\[
\left(\mathbf{I} - \gamma \mathbf{P}\right)\mathbf{V} = \mathbf{R}
\]

Where \(\mathbf{I}\) is the identity matrix. Assuming \(\left(\mathbf{I} - \gamma \mathbf{P}\right)\) is invertible, the solution is:

\[
\mathbf{V} = \left(\mathbf{I} - \gamma \mathbf{P}\right)^{-1}\mathbf{R}
\]

<br>

**Elements of MDP:**
$$M = (S, A, P, R, \gamma)$$

Where:

  S: **Set of states**
  - The set of all possible situations the agent can be in
  
  A: **Set of actions**
  - The set of all possible decisions the agent can make
  
  P: **Transition probability function**
  - $P(s' | s, a)$ = Probability of transitioning to state $s'$ when taking action $a$ in state $s$
  
  R: **Reward function**
  - $R(s, a, s')$ = Immediate reward received after transitioning from state $s$ to state $s'$ due to action $a$
  Sometimes simplified as $R(s)$ or $R(s,a)$

  γ: **Discount factor**
  - $0 \leq \gamma \leq 1$
  - Controls the importance of future rewards relative to immediate rewards
  - $\gamma = 0$: Agent only cares about immediate rewards
  - $\gamma = 1$: Agent prioritize delayed / long term rewards.

<br>

<u>**Link Between MDP and MRP**</u>

- **Markov Decision Process (MDP):**
  - An MDP is a framework for modeling decision-making in environments where outcomes are partly random and partly under the control of an agent.
  - It is defined by a tuple \((S, A, P, R, \gamma)\):
    - **\(S\)**: Set of states.
    - **\(A\)**: Set of actions.
    - **\(P(s' \mid s, a)\)**: Transition probabilities.
    - **\(R(s, a)\)**: Reward function.
    - **\(\gamma\)**: Discount factor.

- **Markov Reward Process (MRP):**
  - An MRP is a special case of an MDP where no decisions are made—only the probabilistic state transitions and rewards remain.
  - It is defined by a tuple \((S, P, R, \gamma)\), essentially removing the action component.

- **The Link:**
  - **Policy Evaluation:** When a policy \(\pi\) is fixed in an MDP, the decision-making is resolved by the policy. The resulting process, where the state transitions and rewards are determined by \(\pi\), is an MRP.
  - **Reduction:** An MDP reduces to an MRP under a fixed policy. In other words, by choosing a specific action in every state (as dictated by the policy), the MDP's dynamics simplify to those of an MRP.
  - **Purpose:** The MRP formulation is used to evaluate the expected return (or value function) of a fixed policy within an MDP.

In summary, an MRP is the result of applying a fixed policy to an MDP, thereby converting the decision process into a process with only states, transitions, and rewards.


---