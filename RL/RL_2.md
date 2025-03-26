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
