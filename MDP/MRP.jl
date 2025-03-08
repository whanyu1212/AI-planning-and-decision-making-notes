using LinearAlgebra
"""
    compute_value_mrp(P::Matrix{Float64}, R::Vector{Float64}, gamma::Float64; theta::Float64=1e-6)::Vector{Float64}

    Compute the state-value function V for a Markov Reward Process using iterative dynamic programming.

    # Arguments
    - `P::Matrix{Float64}`: Transition probability matrix where P[s,s'] is the probability of transitioning from state s to s'
    - `R::Vector{Float64}`: Reward vector where R[s] is the reward for being in state s
    - `gamma::Float64`: Discount factor (between 0 and 1)
    - `theta::Float64=1e-6`: Convergence threshold for determining when to stop iterations

    # Returns
    - `V::Vector{Float64}`: The computed state value function vector
"""
function compute_value_mrp(P::Matrix{Float64}, R::Vector{Float64}, gamma::Float64; theta::Float64=1e-6)::Vector{Float64}
    n = length(R)
    V = zeros(Float64, n)
    
    while true
        delta = 0.0
        V_new = copy(V)
        
        for s in 1:n
            # Bellman update: V(s) = R(s) + gamma * sum(P[s, s'] * V(s'))
            V_new[s] = R[s] + gamma * dot(P[s, :], V)
            delta = max(delta, abs(V[s] - V_new[s]))
        end
        
        V = V_new
        
        # The algorithm has converged enough
        if delta < theta
            break
        end
    end
    
    return V
end

# Example usage
function main()
    # Example 1:
    # P(S1|S1) = 0.5, P(S2|S1) = 0.5
    # P(S1|S2) = 0.2, P(S2|S2) = 0.8
    # R(S1) = 1, R(S2) = 2
    # gamma = 0.9
    P = [0.5 0.5; 0.2 0.8]
    R = [1.0, 2.0]
    gamma = 0.9
    
    V = compute_value_mrp(P, R, gamma)
    println("Value Function: ", V)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end