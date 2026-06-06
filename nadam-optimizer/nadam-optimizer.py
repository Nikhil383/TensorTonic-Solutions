import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    w=np.array(w,dtype=float)
    m=np.array(m,dtype=float)
    v=np.array(v,dtype=float)
    grad=np.array(grad,dtype=float)

    #update the first momement
    m_new=beta1*m+(1-beta1)*grad

    #update the second moment
    v_new=beta2*v+(1-beta2)*(grad**2)

    #nesterov_adjusted update
    lradam=beta1*m_new+(1-beta1)*grad
    w_new=w-lr*lradam/(np.sqrt(v_new)+eps)

    return w_new,m_new,v_new