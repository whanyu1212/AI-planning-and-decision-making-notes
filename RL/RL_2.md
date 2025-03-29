<!-- omit in toc -->
# Reinforcement Learning Part 2
Continuation from Part 1

<!-- omit in toc -->
## Table of Contents
- [Clarifications](#clarifications)
  - [Q(s,a) vs V^π(s)](#qsa-vs-vπs)
  - [On Policy vs Off Policy](#on-policy-vs-off-policy)
- [Scaling Problem](#scaling-problem)
  - [Function Approximation](#function-approximation)
- [Linear Function Approximation](#linear-function-approximation)
  - [Approximating Monte Carlo Learning](#approximating-monte-carlo-learning)
    - [Additional Numerical Example](#additional-numerical-example)
  - [Approximating Temporal Difference Learning](#approximating-temporal-difference-learning)
    - [Numerical Example (Q-Learning with Linear Function Approximation)](#numerical-example-q-learning-with-linear-function-approximation)
  - [Issues with TD Learning](#issues-with-td-learning)
- [Non-Linear Functional Approximation](#non-linear-functional-approximation)
  - [Deep Reinforcement Learning](#deep-reinforcement-learning)
  - [Deep Q-Learning in a nutshell](#deep-q-learning-in-a-nutshell)
- [Policy Search](#policy-search)
  - [Policy Search with Q Function](#policy-search-with-q-function)
  - [Stochastic Policies](#stochastic-policies)
  - [How to Improve the Policy?](#how-to-improve-the-policy)
  - [Policy Gradient](#policy-gradient)
  - [REINFORCE](#reinforce)
- [Advanced Policy Search](#advanced-policy-search)
  - [Trust Region Policy Optimization](#trust-region-policy-optimization)
  - [Additional details of TRPO](#additional-details-of-trpo)
  - [Practical Approximation of TRPO](#practical-approximation-of-trpo)
  - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
  - [Actor Critic Method](#actor-critic-method)
### Clarifications

#### Q(s,a) vs V^π(s)
**\(Q^\pi(s,a)\)** is the **action-value function** under policy \(\pi\). It represents the expected return (sum of discounted future rewards) when starting in state \(s\), taking action \(a\), and then following policy \(\pi\) thereafter. Formally:

\[
Q^\pi(s,a) 
\;=\;
\mathbb{E}\bigl[G_t \mid S_t = s, A_t = a, \text{ thereafter follow } \pi\bigr].
\]

Here’s why it’s often more useful than **\(V^\pi(s)\)** (the state-value function):

1. **Explicitly Captures the Effect of Actions**  
   - \(V^\pi(s)\) tells you the expected return if you start in state \(s\) and follow \(\pi\).  
   - \(Q^\pi(s,a)\) additionally accounts for **which action** \(a\) you choose in state \(s\).  
   - This extra detail is essential for **control** (finding the best action to take).

2. **Needed for Learning an Optimal Policy**  
   - To improve a policy, you need to compare the values of different actions in a given state.  
   - \(Q^\pi(s,a)\) lets you pick the action with the highest value, whereas \(V^\pi(s)\) doesn’t distinguish among actions in the same state.

3. **Basis for \(\varepsilon\)-Greedy Exploration**  
   - Many reinforcement learning algorithms (e.g., SARSA, Q-Learning) rely on \(\varepsilon\)-greedy policies with respect to **action-values**.  
   - You need \(Q^\pi(s,a)\) to select the action with the highest estimated return.

4. **Flexibility**  
   - If you have \(Q^\pi\), you can derive \(V^\pi\) by taking the expectation over the policy’s actions:
     \[
     V^\pi(s) = \sum_a \pi(a \mid s) \, Q^\pi(s,a).
     \]
   - However, going from \(V^\pi\) to \(Q^\pi\) is more complicated without a model of the environment.

In summary, **\(Q^\pi(s,a)\)** provides the value of taking a particular action \(a\) in state \(s\), making it a more direct tool for **control**—that is, for choosing which action to take—than **\(V^\pi(s)\)**, which only gives the value of a state without specifying an action.

<br>

#### On Policy vs Off Policy
- On Policy: evaluates and improves the same policy that is used to make decisions and generate the data trajectories (e.g., SARSA)
- Off Policy: learns about a different (target) policy than the one used to generate the data

### Scaling Problem
- Number of states grow exponentially with the # of features
- Tabular representation only scales to tens of thousands of states
- Action space could also be large
- We can interpolate across states to mitigate redundant information

#### Function Approximation

- **Constructs a compact representation** of the true utility (value) function and Q-function.
- **Example:** Represent an evaluation function for chess as a linear function of features (or basis functions):
  \[
  \hat{V}_\theta(s) \;=\; \theta_1 f_1(s) + \theta_2 f_2(s) + \dots + \theta_n f_n(s).
  \]
- **Parameters \(\theta\)** define a function over a potentially huge state space:
  - Typically, \(\theta \in \mathbb{R}^n\) or \(\mathbb{R}^{n+1}\), with \(\lvert S \rvert \gg n\).
- **Goal:** The RL agent learns \(\theta\) that best approximates the evaluation (utility) function.

- **Generalization**: Allows learning from a **small number of observed states** to cover the **entire state space**.
  - **Example**: A Backgammon agent learned to play as well as the best human players by observing only \(\approx 10^{12}\) states out of \(10^{20}\) possible states.

- **Caveats**:
  - **Poor Choice of Features** can lead to:
    - **State Aliasing**: Multiple distinct states might map to the same feature vector, causing confusion.
    - **Insufficient Model Capacity**: If \(n\) (the number of parameters) is too small, the model might fail to achieve a good approximation.
  - **Feature Design**:
    - Where do the features \(f\) come from (e.g., in a game of chess)?  
    - Often, we assume they exist or are engineered beforehand.


### Linear Function Approximation

<u>**Key Idea:**</u>

Use a linear function to approximate \(V\):

\[
\hat{V}_\theta(s) \;=\; \theta_0 + \theta_1\,f_1(s) \;+\; \theta_2\,f_2(s) \;+\; \dots \;+\; \theta_N\,f_N(s),
\]

where \(N\) is the number of features.

In vector form:

\[
\hat{V}_\theta(s) \;=\; \theta^T\, f(s).
\]

We employ **homogenization** (adding a bias term: \( \hat{V}_\theta(s) = \theta_0 \cdot 1 + \sum_{i=1}^N \theta_i f_i(s) = \theta^T f(s) \)) by setting one of the features to 1.

#### Approximating Monte Carlo Learning
- **For MC learning**, we get a set of training samples \(\bigl((x_1, y_1), u_1\bigr), \bigl((x_2, y_2), u_2\bigr), \dots, \bigl((x_n, y_n), u_n\bigr)\),
  where \(u_j\) is the return of the \(j\)-th episode.

- **This is a supervised learning problem**:
  - We have input features (e.g., \((x_i, y_i)\)) and target values (the returns \(u_i\)).

- **Standard linear regression problem with mean squared error**:
  - The error is minimized when the partial derivatives w.r.t. the coefficients of the linear function are zero.

<br>

**Learning While Interacting with the Environment**
- Update the parameters after each trial (episode or sample).

**Loss Function**
\[
\mathcal{E}_j(s) \;=\; \tfrac{1}{2}\,\bigl[\hat{V}_\theta(s) - u_j(s)\bigr]^2
\]
where \(u_j(s)\) is the return from state \(s\) in the \(j\)-th trial.

**Update Rule**
\[
\theta_i \;\leftarrow\; \theta_i \;-\; \alpha \,\frac{\partial \,\mathcal{E}_j(s)}{\partial \,\theta_i}
\]
Substituting the derivative:
\[
\theta_i 
\;\leftarrow\; 
\theta_i 
\;-\; 
\alpha\,\bigl[\hat{V}_\theta(s) - u_j(s)\bigr]\,
\frac{\partial \,\hat{V}_\theta(s)}{\partial \,\theta_i}
\]
Often rewritten as:
\[
\theta_i 
\;\leftarrow\; 
\theta_i 
\;+\; 
\alpha\,\bigl[u_j(s) - \hat{V}_\theta(s)\bigr]\,
\frac{\partial \,\hat{V}_\theta(s)}{\partial \,\theta_i}
\]

<br>

**Interpretation:**
- \( \mathcal{E}_j(s) \) measures the squared difference between the agent’s predicted value \( \hat{V}_\theta(s) \) and the observed return \( u_j(s) \).

- If the prediction \(\hat{V}_\theta(s)\) is too low compared to \(u_j(s)\), we **increase** \(\theta_i\).

- If it’s too high, we **decrease** \(\theta_i\).

- \(\alpha\) (the learning rate) controls how big each update step is.

<br>

<u>*One gradient step or not?*</u>
It depends on the learning approach:

- **Online / Incremental Learning:**  
  Typically, you take one gradient step per sample (or per update). For example, in TD(0) or on-policy Monte Carlo control, you update \(\theta\) once for each observed sample.

- **Batch / Offline Learning:**  
  In this setting, you can perform as many gradient steps as needed on your batch of data until convergence or until a stopping criterion is met.

In many reinforcement learning applications, especially when learning in an online fashion, one gradient step per sample is common.


**Example**

<u>*Recall:*</u>  
\[
\hat{V}_\theta(x, y) \;=\; \theta_0 + \theta_1\,x \;+\; \theta_2\,y.
\]  

If \(\theta_0 = 0.5\), \(\theta_1 = 0.2\), \(\theta_2 = 0.1\), then  
\[
V(1,1) = 0.5 + 0.2 \cdot 1 + 0.1 \cdot 1 = 0.8.
\]

---

<u>*Update Rule:*</u>

Given a state \(s = (x,y)\), the gradient-descent update rule is:

\[
\theta_i \;\leftarrow\; \theta_i \;-\; \alpha \,\frac{\partial \,\mathcal{E}_j(s)}{\partial \,\theta_i}
\quad\Longrightarrow\quad
\theta_i \;\leftarrow\; \theta_i \;+\; \alpha\,\bigl[u_j(s) - \hat{V}_\theta(s)\bigr]\,
\frac{\partial \,\hat{V}_\theta(s)}{\partial \,\theta_i}.
\]

Concretely, for this linear function:

\[
\hat{V}_\theta(x,y) = \theta_0 + \theta_1\,x + \theta_2\,y,
\]
the partial derivatives are:

$$
\frac{\partial \,\hat{V}_\theta(x,y)}{\partial \,\theta_0} = 1
$$

$$
\frac{\partial \,\hat{V}_\theta(x,y)}{\partial \,\theta_1} = x
$$

$$
\frac{\partial \,\hat{V}_\theta(x,y)}{\partial \,\theta_2} = y
$$

Thus, the updates become:
\[
\theta_0 \;\leftarrow\; \theta_0 \;+\; \alpha\,\bigl[u_j(s) - \hat{V}_\theta(s)\bigr],
\]
\[
\theta_1 \;\leftarrow\; \theta_1 \;+\; \alpha\,\bigl[u_j(s) - \hat{V}_\theta(s)\bigr]\,x,
\]
\[
\theta_2 \;\leftarrow\; \theta_2 \;+\; \alpha\,\bigl[u_j(s) - \hat{V}_\theta(s)\bigr]\,y.
\]


##### Additional Numerical Example

Suppose:
- Initial parameters: \(\theta_0 = 0.5\), \(\theta_1 = 0.2\), \(\theta_2 = 0.1\).
- State \(s = (x=2, y=3)\).
- Observed return \(u_j(s) = 2.0\).
- Learning rate \(\alpha = 0.1\).

1. **Compute Current Prediction**  
   \[
   \hat{V}_\theta(2,3) = 0.5 + 0.2 \cdot 2 + 0.1 \cdot 3 = 0.5 + 0.4 + 0.3 = 1.2.
   \]

2. **Calculate the Error**  
   \[
   \delta = u_j(s) - \hat{V}_\theta(s) = 2.0 - 1.2 = 0.8.
   \]

3. **Update \(\theta_0\)**  
   \[
   \theta_0 \;\leftarrow\; 0.5 + 0.1 \cdot 0.8 = 0.5 + 0.08 = 0.58.
   \]

4. **Update \(\theta_1\)**  
   \[
   \theta_1 \;\leftarrow\; 0.2 + 0.1 \cdot 0.8 \cdot 2 = 0.2 + 0.16 = 0.36.
   \]

5. **Update \(\theta_2\)**  
   \[
   \theta_2 \;\leftarrow\; 0.1 + 0.1 \cdot 0.8 \cdot 3 = 0.1 + 0.24 = 0.34.
   \]

**Result:**  
- After one update step, the parameters become \(\theta_0 = 0.58\), \(\theta_1 = 0.36\), and \(\theta_2 = 0.34\).  
- This shifts the predicted value closer to the observed return.

**Updated Prediction:**  
\[
\hat{V}_\theta(2,3) = 0.58 + 0.36 \cdot 2 + 0.34 \cdot 3 
= 0.58 + 0.72 + 1.02 
= 2.32.
\]
It has now overshot a bit, but subsequent updates (from additional samples) will gradually refine \(\theta\) toward a good approximation.

---

#### Approximating Temporal Difference Learning

- For TD learning, the same idea of online learning can be applied:
  - Adjust the parameters to reduce the temporal difference between successive states.

- **For utilities**:  
  \[
  \theta_i \;\leftarrow\; \theta_i + \alpha \,\bigl[R(s,a,s') + \gamma\,\hat{V}_\theta(s') - \hat{V}_\theta(s)\bigr]\;\frac{\partial\,\hat{V}_\theta(s)}{\partial\,\theta_i}
  \]  
  (**TD target**: \(R(s,a,s') + \gamma\,\hat{V}_\theta(s')\))

- **For Q-learning**:  
  \[
  \theta_i \;\leftarrow\; \theta_i + \alpha \,\Bigl[R(s,a,s') + \gamma\,\max_{a}\,\hat{Q}_\theta(s',a) - \hat{Q}_\theta(s,a)\Bigr]\;\frac{\partial\,\hat{Q}_\theta(s,a)}{\partial\,\theta_i}
  \]  
  (**TD target**: \(R(s,a,s') + \gamma\,\max_{a}\,\hat{Q}_\theta(s',a)\))

- **Notes**:
  - Also called **semi-gradient methods**: the target is *not* a true value; it depends on \(\theta\).
  - For **prediction** problems, the update rule converges for linear function approximation when on-policy.


##### Numerical Example (Q-Learning with Linear Function Approximation)

Suppose we approximate the action-value function as  
\( \hat{Q}_\theta(s,a) = \theta^\top x(s,a) \),  
where \( x(s,a) \) is a feature vector for the state–action pair. Below is a single-step Q-learning update example:

1. **Initial Parameters**  
   - Parameter vector: \(\theta = (0.1,\, 0.2,\, 0.3)\)  
   - Learning rate: \(\alpha = 0.1\)  
   - Discount factor: \(\gamma = 0.9\)

2. **Current State–Action**  
   - Let the current state–action pair be \((s, a)\).  
   - Feature vector: \( x(s, a) = (1,\, 2,\, 3) \).  
   - Predicted Q-value:  
     \( \hat{Q}_\theta(s,a) = 0.1 \times 1 + 0.2 \times 2 + 0.3 \times 3 = 1.4 \).

3. **Next State**  
   - After taking action \(a\) in state \(s\), observe:
     - Reward \( R = 1 \).
     - Next state \( s' \).
   - For Q-learning, we use \( \max_{a'} \hat{Q}_\theta(s', a') \) to form the TD target.  
   - Suppose the best next action \( a' \) has feature vector \( x(s', a') = (1,\, 0,\, 1) \). Then:  
     \( \hat{Q}_\theta(s', a') = 0.1 \times 1 + 0.2 \times 0 + 0.3 \times 1 = 0.4 \).

4. **TD Target**  
   \( \text{Target} = R + \gamma \times \max_{a'} \hat{Q}_\theta(s', a') = 1 + 0.9 \times 0.4 = 1.36 \).

5. **TD Error**  
   \( \delta = \text{Target} - \hat{Q}_\theta(s,a) = 1.36 - 1.4 = -0.04 \).

6. **Parameter Update**  
   Since \( \frac{\partial \,\hat{Q}_\theta(s,a)}{\partial \,\theta} = x(s,a) \), the update is:  
   \( \theta \leftarrow \theta + \alpha \,\delta \, x(s,a) \).  

   Substituting the values:  
   \( \theta \leftarrow (0.1,\, 0.2,\, 0.3) + 0.1 \times (-0.04) \times (1,\, 2,\, 3). \)

   - \( \theta_0 \leftarrow 0.1 + 0.1 \times (-0.04) \times 1 = 0.1 - 0.004 = 0.096 \)
   - \( \theta_1 \leftarrow 0.2 + 0.1 \times (-0.04) \times 2 = 0.2 - 0.008 = 0.192 \)
   - \( \theta_2 \leftarrow 0.3 + 0.1 \times (-0.04) \times 3 = 0.3 - 0.012 = 0.288 \)

7. **New Parameters**  
   \( \theta = (0.096,\, 0.192,\, 0.288). \)

8. **Interpretation**  
   - The TD target was slightly less than the current prediction, so all parameters decreased a bit.  
   - With more samples and updates, \(\theta\) will move toward values that reduce the TD error on average.

#### Issues with TD Learning

- **The Deadly Triad**
  - *Definition*: The combination of function approximation, bootstrapping, and off-policy training.
  - *Why It Causes Issues*:  
    - **Function Approximation**: Generalizes across states but can introduce inaccuracies.
    - **Bootstrapping**: Updates estimates based on other estimates, which can propagate and amplify errors.
    - **Off-Policy Training**: Learning about one policy while following another may lead to using unrepresentative data.
  - *Result*: This combination can lead to instability or divergence in learning.

- **Catastrophic Forgetting**
  - *Definition*: The phenomenon where a learning system forgets previously learned information when updating with new data.
  - *Why It Happens*:  
    - The model overfits to recent experiences, which can override and erase earlier learned knowledge.
  - *Result*: The policy might perform well on recent tasks but poorly on situations encountered earlier in training.

- **Issues with Discounting**
  - *Traditional Use*: A discount factor \(\gamma\) (with \(0 \le \gamma < 1\)) prioritizes immediate rewards over future rewards.
  - *In Continual Learning*:  
    - Strict discounting can undervalue long-term rewards in non-episodic, ongoing tasks.
    - \(\gamma\) may need to be tuned as a parameter that balances short-term and long-term rewards, rather than acting as a fixed rate.
  - *Result*: Discounting might be modified or "deprecated" in continual learning scenarios to better reflect the desired trade-off between immediate and future rewards.

---

### Non-Linear Functional Approximation

#### Deep Reinforcement Learning
Absolutely. In Deep Reinforcement Learning (DRL), **neural networks** serve as **function approximators** for either the value function (utilities) or the action-value function (Q). Essentially:

- **Utilities (\(\hat{V}_\theta(s)\))**:  
  The neural network approximates the value of each state \(s\).

- **Q-values (\(\hat{Q}_\theta(s,a)\))**:  
  The network approximates how good it is to take action \(a\) in state \(s\).

In both cases, the parameters \(\theta\) are the weights of the neural network, and you update them (via backpropagation) based on temporal-difference errors.

---

#### Deep Q-Learning in a nutshell
<img src="/RL/1743176594646.jpg" alt="Deep Q-Learning architecture diagram showing how neural networks are used to approximate Q-values" width="500" />

<u>**Variants of Deep Q Learning**</u>

- **DQN (Reference)**  
  - **Target:**  
    \(
    R(s, a, s') \;+\; \gamma \,\max_{a'}\,Q\bigl(s',\,a';\,\theta^-\bigr)
    \)
  - This is the basic Deep Q-Network algorithm, which introduced experience replay and a target network.

- **Double DQN** (see S.B. for detail)  
  - Aimed at reducing the **overestimation** issue in standard DQN.  
  - **Target:**  
    \(
    R(s, a, s') \;+\; \gamma\,Q\Bigl(s',\,\arg\max_{a'}\,Q\bigl(s',\,a';\,\theta\bigr);\;\theta^-\Bigr)
    \)

- **Multi-Step Learning (n-step SARSA / Q-learning)**  
  - Uses multiple steps of returns before bootstrapping.  
  - **Target (example for n-step):**  
    \(
    r_1 + r_2 + \dots + \gamma^n\,Q\bigl(s_n,\,a_n;\,\theta^-\bigr)
    \)

- **Distributional RL**  
  - Learns a **probability distribution** of returns instead of a single scalar value.  
  - This can capture different levels of uncertainty or risk in the returns.

- **Prioritized Replay**  
  - **Prioritizes** sampling of experiences with **high TD error** from the replay buffer, focusing learning on more “surprising” or important transitions.


### Policy Search

- **A policy** is a mapping from a state to an action:
  \[
  \pi: S \;\to\; A
  \]

- We can **parameterize the policy** with parameters \(\theta\): 
  \[
  \pi_\theta
  \]

- **Policy search**: Adjust \(\theta\) to improve the policy.
  - Keep “twiddling” the policy as long as its performance improves.
  - In the **tabular case**, “twiddling” means becoming greedy with respect to the estimated value.

#### Policy Search with Q Function

- **Greedy Policy from Q-Function:**
  - Once you have a parameterized Q-function, \(\hat{Q}_\theta\), the policy is derived by choosing the action that maximizes it:
    \[
    \pi_\theta(s) = \arg\max_a \, \hat{Q}_\theta(s,a).
    \]

- **Difference from Q-Learning:**
  - **Q-Learning (with function approximation):**
    - Aims to have \(\hat{Q}_\theta\) closely match the true optimal Q-values \(Q^*\).
  - **Policy Search:**
    - Only requires that \(\hat{Q}_\theta\) yields the correct argmax (i.e., the best action), not that its numerical estimates are accurate.
    
- **Not Unique:**
  - Scaling or shifting \(\hat{Q}_\theta\) does not affect the argmax.
  - Therefore, even if \(\hat{Q}_\theta\) isn’t numerically close to \(Q^*\), it can still induce the same greedy policy.

- **Issues with Discrete Actions:**
  - **Discontinuity:**
    - In a discrete action space, the policy \(\pi_\theta(s) = \arg\max_a \, \hat{Q}_\theta(s,a)\) is a discontinuous function of \(\theta\). Small changes in \(\theta\) can cause abrupt changes in the selected action.
  - **Gradient Challenges:**
    - Because the action selection changes abruptly, the policy is not smooth with respect to \(\theta\), making standard gradient-based optimization methods difficult or unstable.
  - **Drastic Changes:**
    - A small tweak in \(\theta\) can suddenly change the entire policy, leading to large swings in performance rather than gradual improvement.

#### Stochastic Policies

- **Randomized Policy \(\pi_\theta(s,a)\)**
  - A **deterministic** policy is \(\pi_\theta(s) = \arg\max_a \hat{Q}_\theta(s,a)\).  
  - This is a special case of a randomized policy:
    \[
    \pi_\theta(s,a) = 
    \begin{cases}
      1 & \text{if } a = \arg\max_{a'} \hat{Q}_\theta(s,a'),\\
      0 & \text{otherwise.}
    \end{cases}
    \]
- **Parameterizing Randomized Policies**
  - Often done with a **softmax** over \(\hat{Q}_\theta\):
    \[
    \pi_\theta(s,a) = \frac{e^{\,\beta\,\hat{Q}_\theta(s,a)}}{\sum_{a'} e^{\,\beta\,\hat{Q}_\theta(s,a')}}.
    \]
  - \(\beta\) modulates how “greedy” the policy is (higher \(\beta\) \(\to\) closer to \(\arg\max\)).
  - This distribution is **differentiable** w.r.t. \(\theta\), allowing gradient-based updates.

---

#### How to Improve the Policy?

- **Policy Value** \(\rho(\theta)\)
  - \(\rho(\theta)\) is the expected return when \(\pi_\theta\) is executed.
- **Assumption**: \(\rho(\theta)\) is differentiable.
- **Deterministic Policy & Environment**  
  - Taking the gradient \(\nabla_\theta \rho(\theta)\) is problematic if the policy is purely \(\arg\max\) (non-differentiable).
  - Small changes in \(\theta\) can cause large, discontinuous jumps in the chosen action.
- **Stochastic Policy & Environment**  
  - We can obtain an **estimate** of \(\nabla_\theta \rho(\theta)\) from sampled trajectories.
  - Then apply **gradient ascent** to improve the policy parameters \(\theta\).

---

#### Policy Gradient

1. **Single Action from a Single State \(S_0\)**
   - Consider the expected reward when we take different actions from \(S_0\):
     \[
     \nabla_\theta \rho(\theta) 
       = \nabla_\theta \sum_a R(S_0, a, S_0)\,\pi_\theta(S_0,a)
       = \sum_a R(S_0, a, S_0)\,\nabla_\theta \pi_\theta(S_0,a).
     \]
   - **Approximation with Samples**:  
     If we generate samples \(\{a_j\}\) by drawing actions from \(\pi_\theta(S_0,\cdot)\), then
     \[
     \nabla_\theta \rho(\theta) 
       \approx \sum_{a} \pi_\theta(S_0,a)\,\nabla_\theta \pi_\theta(S_0,a)
       = \frac{1}{N}\sum_{j=1}^N \ldots
     \]
     (depending on how we collect returns and handle the gradient).

2. **For a Single Case**  
   - One common form is:
     \[
     \nabla_\theta \rho(\theta) 
       \approx \frac{1}{N}\sum_{j=1}^N 
         \frac{R(S_0,a_j,S_0)\,\nabla_\theta \pi_\theta(S_0,a_j)}{\pi_\theta(S_0,a_j)},
     \]
     where \(a_j\) is the action sampled in the \(j\)-th trial.

3. **Sequential Case (Multiple States)**  
   - Generalizes to:
     \[
     \nabla_\theta \rho(\theta) 
       \approx \frac{1}{N}\sum_{j=1}^N 
         \sum_{s \,\in\, \text{trajectory}_j} 
         \frac{u_j(s)\,\nabla_\theta \pi_\theta(s,a_j)}{\pi_\theta(s,a_j)},
     \]
     where \(u_j(s)\) is the total return from state \(s\) onward in the \(j\)-th trajectory, and \(\pi_\theta(s,a_j)\) is the probability of choosing action \(a_j\) in state \(s\) under the current policy.

- **Interpretation**: We **scale** the gradient of the log-probability of actions by the returns they produced. Over many samples, this pushes \(\theta\) to favor actions that yield higher returns.

#### REINFORCE

- **Monte-Carlo Policy Gradient**  
  - Uses entire episode returns to update the policy parameters.

- **Online Update**  
  - The REINFORCE algorithm (for episode \(j\)) can be written as:
    \[
    \theta_{j+1} 
    = \theta_j + \alpha\,u_j\,\nabla_\theta \pi_\theta(s,a),
    \]
    where \(u_j\) is the return observed from action \(a\) in state \(s\).

- **Using the Identity**  
  \[
  \nabla_\theta \ln \,\pi_\theta(s,a)
  = \frac{\nabla_\theta \pi_\theta(s,a)}{\pi_\theta(s,a)},
  \]
  we can rewrite the update in terms of the **log** of the policy:

- **Rewriting**  
  \[
  \theta_{j+1} 
  = \theta_j + \alpha \, u_j \, \nabla_\theta \ln \,\pi_\theta(s,a).
  \]

### Advanced Policy Search

#### Trust Region Policy Optimization

<u>**Motivation:**</u>

- **Standard Policy Gradient Pitfall**  
  If you take a large gradient step, the new policy might be too different from the old one. This can cause large drops in performance (e.g., overstepping into a region of parameter space that’s far worse).

- **Idea: Trust Region**  
  Enforce a “trust region” around the old policy so each update doesn’t move the policy too far. This helps maintain monotonic improvements (or at least reduces the chance of catastrophic drops).

- **Problem**: The new policy \(\pi_\theta\) can be very far from the old policy \(\pi_{\text{old}}\), causing large performance drops.
- **Solution**: Maximize the objective **within a trust region**:
  \[
  \text{maximize } L(\pi_\theta) \quad (\text{approximates } \rho(\theta))
  \]
  **subject to**
  \[
  \max_s \, \text{KL}\bigl(\pi_{\text{old}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\bigr) \;\le\; \delta.
  \]
- **Interpretation**: Take “small steps” so that \(\pi_\theta\) doesn’t deviate too much (in KL divergence) from \(\pi_{\text{old}}\).

#### Additional details of TRPO

- **Goal**  
  Find a policy \(\pi_\theta\) that **maximizes** the expected return:
  \[
  \rho(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\bigl[\sum_{t=0}^\infty \gamma^t r_t\bigr].
  \]

- **Data Collection**  
  - We collect data (states, actions, rewards) using the **old policy** \(\pi_{\theta_{\text{old}}}\).
  - This makes it useful to rewrite the **new policy’s objective** in terms of \(\pi_{\theta_{\text{old}}}\).

- **Advantage Function**  
  \[
  A^{\pi_{\theta_{\text{old}}}}(s,a) 
    = \mathbb{E}\bigl[r_t + \gamma\,V^{\pi_{\theta_{\text{old}}}}(s_{t+1}) - V^{\pi_{\theta_{\text{old}}}}(s)\bigr],
  \]
  which measures how much better (or worse) it is to take action \(a\) in state \(s\) compared to the old policy’s value.

- **Rewriting the Objective**  
  - If \(\rho_{\pi_{\theta_{\text{old}}}}(s)\) represents the **discounted visitation frequency** of state \(s\) under \(\pi_{\theta_{\text{old}}}\), then
    \[
    V(\pi_\theta) 
    = V\bigl(\pi_{\theta_{\text{old}}}\bigr) 
      + \sum_s \rho_{\pi_{\theta_{\text{old}}}}(s) 
      \sum_a \pi_\theta(a \mid s)\,A^{\pi_{\theta_{\text{old}}}}(s,a).
    \]
  - This exact form can be **hard to optimize** directly, so TRPO uses a **surrogate objective**:
    \[
    L(\pi_\theta) 
    \approx 
    \sum_s \rho_{\pi_{\theta_{\text{old}}}}(s) 
    \sum_a \pi_\theta(a \mid s)\,A^{\pi_{\theta_{\text{old}}}}(s,a).
    \]

---

#### Practical Approximation of TRPO

- **Sampled Version** of \(L(\pi_\theta)\)  
  - Instead of summing over all states \(s\), we sample states and actions from **trajectories** generated by \(\pi_{\theta_{\text{old}}}\).  
  - This gives an empirical estimate:
    \[
    L(\pi_\theta) 
    \approx 
    \frac{1}{N} \sum_{i=1}^N 
      \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)} 
      \,A^{\pi_{\theta_{\text{old}}}}(s_i, a_i).
    \]

- **Expected KL Divergence**  
  - Instead of bounding the **maximum** KL divergence, TRPO often uses the **expected** KL divergence across sampled states:
    \[
    \mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}}
      \bigl[
        D_{\text{KL}}\bigl(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s)\bigr)
      \bigr]
    \;\le\; \delta.
    \]
  - This is easier to estimate from data and still maintains a **trust region** that prevents overly large policy updates.

#### Proximal Policy Optimization (PPO)

- **Motivation**  
  - PPO refines the TRPO approach by **simplifying** the objective function and implementation, while still limiting large policy updates.

- **Basic Idea**  
  - Define a **probability ratio** for the new policy \(\pi_\theta\) relative to the old policy \(\pi_{\theta_{\text{old}}}\):
    \[
    \mu_i(\pi_\theta) 
      = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)}.
    \]
  - Then, a naive objective might be:
    \[
    \sum_i \mu_i(\pi_\theta)\,A^{\pi_{\text{old}}}(s_i,a_i),
    \]
    which is the same surrogate objective used in TRPO (but without a KL constraint).

- **Clipped Objective**  
  - PPO **clips** the probability ratio to stay within \([1-\varepsilon,\,1+\varepsilon]\) (for some small \(\varepsilon\), e.g., 0.1 or 0.2):
    \[
    \sum_i \min\Bigl(
      \mu_i(\pi_\theta)\,A^{\pi_{\text{old}}}(s_i,a_i), 
      \text{clip}\bigl(\mu_i(\pi_\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,A^{\pi_{\text{old}}}(s_i,a_i)
    \Bigr).
    \]
  - **Interpretation**:  
    - If \(\mu_i(\pi_\theta)\) stays in the range \([1-\varepsilon,\,1+\varepsilon]\), it’s left untouched.  
    - If it goes outside that range, the objective is clipped, preventing **excessive** deviation from the old policy.

- **Why Clipping?**  
  - By **limiting** how far \(\mu_i(\pi_\theta)\) can deviate from 1 (the old policy), PPO **prevents overly large policy updates**.  
  - This serves a similar purpose as the KL constraint in TRPO but is typically **simpler** to implement and tune.

- **Outcome**  
  - PPO often achieves **stable** and **efficient** policy improvement, balancing **exploration** and **exploitation** without the complexity of a full trust region optimization.

<img src="/RL/1743247446544.jpg" alt="PPO" width="500" />

With the **clipped objective**, the policy stops improving the objective if

$\mu_i(\pi) = \frac{\pi_\theta(a_i \mid s_i)}{\pi_{\theta_{\text{old}}}(a_i \mid s_i)}$


is too far above 1, thereby **discouraging excessively large policy updates**.

#### Actor Critic Method

- **Combined approach**: Integrates value function approximation with policy search
- **Dual parameter updates**:
  - Policy parameters (θ) for the actor
  - Value function parameters (w) for the critic
- **Component roles**:
  - **Actor**: Learns a policy π_θ(s,a) that selects actions
  - **Critic**: Learns a value function V_w(s) or Q_w(s,a) that evaluates those actions
- **Workflow**:
  1. Actor proposes actions based on current policy
  2. Critic evaluates those actions using learned value estimates
  3. Actor improves policy using critic's feedback
  4. Critic refines its own value estimates from observed returns

This approach combines the strengths of both value-based methods and policy gradient methods while mitigating their individual weaknesses.


- **REINFORCE**: Uses a Monte Carlo estimate of the advantage function, which can lead to higher variance.

- **Advantage Function (TD Method)**
  For the TD method, the advantage function can be written as:
  $$
  Q_{\pi_\theta}(s,a) - V_{\pi_\theta}(s) = r + \gamma\,V_{\pi_\theta}(s') - V_{\pi_\theta}(s)
  $$

- **Value Function Estimator**
  Suppose we have a parametric value function $\hat{V}(s, w)$. Then the **TD-type update** for the actor parameters $\theta$ is:
  $$
  \theta_{j+1} = \theta_j + \alpha \,\nabla_\theta \ln \pi_\theta(s_j, a_j)\,\bigl[r_j + \gamma\,\hat{V}(s_{j+1}, w) - \hat{V}(s_j, w)\bigr]
  $$
  The term $\bigl[r_j + \gamma\,\hat{V}(s_{j+1}, w) - \hat{V}(s_j, w)\bigr]$ serves as an estimate of the **advantage**.

- **Multiple Steps of Rewards**
  It is common to use multiple steps of returns instead of a single step, for example:
  $$
  r_j + \gamma\,r_{j+1} + \gamma^2\,r_{j+2} + \dots + \gamma^k\,r_{j+k}
  $$
