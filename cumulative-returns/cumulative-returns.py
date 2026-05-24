def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    result = []
    
    wealth_factor = 1.0
    
    for r in returns:
        
        wealth_factor *= (1 + r)
        
        cumulative_return = wealth_factor - 1
        
        result.append(cumulative_return)
    
    return result
    
    