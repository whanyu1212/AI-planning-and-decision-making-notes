# Markov Decision Process

### **Value Iteration on a Linear Grid (1D Gridworld)**

#### **Problem Setup**
We have a **1D Gridworld** with **5 states**:

[S] – (1) – (2) – (3) – [G]

- **S** = Start state (0)
- **G** = Goal state (4)
- **Agent can move Left (L) or Right (R)**
- **Rewards**:
  - Each step gives a reward of **-1**
  - Reaching **G** gives a reward of **0**
- **Discount factor** \( \gamma = 0.9 \)

---

#### **Step 1: Initialize Value Function**
Initially, set all state values to **0**:

| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | 0 | 0 | 0 | 0 | 0 |

---

#### **Step 2: Apply Value Iteration (Bellman Equation)**

\[
V(s) = \max \left( R + \gamma V(s_{\text{left}}), R + \gamma V(s_{\text{right}}) \right)
\]

- Example for state **2**:

\[
V(2) = \max \left( -1 + 0.9 V(1), -1 + 0.9 V(3) \right)
\]

---

#### **Step 3: Iterative Updates**
##### **Iteration 1**
Using \( V(s) = 0 \):

| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | -1 | -1 | -1 | -1 | 0 |

---

##### **Iteration 2**
| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | -1.9 | -1.9 | -1.9 | -1 | 0 |

---

##### **Iteration 3**
| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | -2.71 | -2.71 | -2.52 | -1.9 | 0 |

---

##### **Final Values After Convergence**
| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | **-3.8** | -3.4 | -2.9 | -2.4 | **0** |

---

##### **Step 4: Extract the Optimal Policy**
We extract the **optimal policy** by choosing the action that leads to the highest value.

| State | 0 (S) | 1 | 2 | 3 | 4 (G) |
|:-----:|:-----:|:---:|:---:|:---:|:-----:|
| \( V(s) \) | -3.8 | -3.4 | -2.9 | -2.4 | **0** |
| **Best Action \( \pi(s) \)** | **Right →** | **Right →** | **Right →** | **Right →** | **Goal 🎯** |

---

#### **Summary**
1. **Initialize \( V(s) = 0 \)**.
2. **Iteratively update values using the Bellman equation**.
3. **Stop when values converge**.
4. **Extract the best policy by following the highest values**.

👉 **Value Iteration helps the agent learn the best way to reach the goal by improving value estimates iteratively!**