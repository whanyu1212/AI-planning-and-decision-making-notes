using LinearAlgebra

"""
    computevalueMDP(P::Array{Float64,3}, R::Array{Float64,3}, gamma::Float64; theta::Float64=1e-6)

Performs value iteration for a Markov Decision Process (MDP).

# Arguments
- `P::Array{Float64,3}`: Transition probability array where P[s,a,s'] is the probability 
  of transitioning from state s to state s' when taking action a
- `R::Array{Float64,3}`: Reward array where R[s,a,s'] is the reward received when 
  transitioning from state s to state s' by taking action a
- `gamma::Float64`: Discount factor (between 0 and 1)
- `theta::Float64=1e-6`: Convergence threshold for determining when to stop iterations

# Returns
- `V::Vector{Float64}`: Optimal value function vector
- `policy::Vector{Int}`: Optimal policy vector with action indices
"""
function computevalueMDP(P::Array{Float64,3}, R::Array{Float64,3}, gamma::Float64; theta::Float64=1e-6)
    n, m, _ = size(P)
    V = zeros(Float64, n)
    policy = zeros(Int, n)
    
    while true
        delta = 0.0
        
        for s in 1:n
            # Compute Q(s,a) for all actions a in state s
            Q_sa = zeros(Float64, m)
            
            for a in 1:m
                # Expected value of taking action a in state s
                # sum of P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])
                Q_sa[a] = sum(P[s,a,:] .* (R[s,a,:] .+ gamma .* V))
            end
            
            # Find the maximum Q(s,a) and the corresponding action
            max_val = maximum(Q_sa)
            best_action = argmax(Q_sa)
            
            # Update delta for convergence check
            delta = max(delta, abs(max_val - V[s]))
            
            # Update value function and policy
            V[s] = max_val
            policy[s] = best_action
        end
        
        # Check for convergence
        if delta < theta
            break
        end
    end
    
    return V, policy
end

# Example usage
function main()
    # Define a simple MDP with 3 states and 2 actions
    P = zeros(Float64, 3, 2, 3)
    
    # Transitions for state 1 (index 1 in Julia)
    P[1,1,:] = [0.8, 0.2, 0.0]  # Action 1
    P[1,2,:] = [0.1, 0.9, 0.0]  # Action 2
    
    # Transitions for state 2 (index 2 in Julia)
    P[2,1,:] = [0.0, 0.6, 0.4]  # Action 1
    P[2,2,:] = [0.0, 0.3, 0.7]  # Action 2
    
    # Transitions for state 3 (index 3 in Julia) - terminal state
    P[3,1,:] = [0.0, 0.0, 1.0]  # Action 1
    P[3,2,:] = [0.0, 0.0, 1.0]  # Action 2
    
    # Reward function
    R = zeros(Float64, 3, 2, 3)
    
    # Rewards for state 1
    R[1,1,:] = [5.0, 10.0, 0.0]  # Action 1
    R[1,2,:] = [3.0, 8.0, 0.0]   # Action 2
    
    # Rewards for state 2
    R[2,1,:] = [0.0, 2.0, 4.0]   # Action 1
    R[2,2,:] = [0.0, 1.0, 7.0]   # Action 2
    
    # Rewards for state 3 (all zeros for terminal state)
    R[3,1,:] = [0.0, 0.0, 0.0]   # Action 1
    R[3,2,:] = [0.0, 0.0, 0.0]   # Action 2
    
    gamma = 0.9
    
    V, policy = computevalueMDP(P, R, gamma)
    println("Optimal Value Function: ", V)
    println("Optimal Policy: ", policy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end