def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    output=[]

    # Loop over states
    for s in range(len(values)):
        best = float('-inf')

        # Loop over actions
        for a in range(len(transitions[s])):
            
            # Compute Q(s, a)
            q = rewards[s][a]

            # Sum over next states
            for s_next in range(len(values)):
                q += gamma * transitions[s][a][s_next] * values[s_next]

            # Track maximum Q-value
            best = max(best, q)

        output.append(best)

    return output
        