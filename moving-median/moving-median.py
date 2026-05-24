def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here
    result = []
    
    for i in range(len(values) - window_size + 1):
        
        # Get current window
        window = values[i:i + window_size]
        
        # Sort window
        sorted_window = sorted(window)
        
        n = len(sorted_window)
        
        # Compute median
        if n % 2 == 1:
            median = sorted_window[n // 2]
        else:
            median = (
                sorted_window[n // 2 - 1] +
                sorted_window[n // 2]
            ) / 2
        
        result.append(median)
    
    return result