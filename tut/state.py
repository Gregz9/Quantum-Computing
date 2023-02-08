from __future__ import annotations 
import numpy as np
import math 
import tensor


class State(tensor.Tensor): 
    """class State represents single and multiple-qubit states.""" 

    def __repr__(self) -> str: 
        s = 'State('
        s += super().__str__().replace('\n', '\n' + ' ' *len(s))
        s += ')'

        return s

    def __str__(self) -> str: 
        s = f'{self.nbits}-qubit state.'
        s += ' Tensor:\n'
        s += super().__str__()
        return s

def qubit(alpha: Optional[np.complexfloating] = None,
          beta: Optional[np.complexfloating] = None) -> State: 

    if alpha is None and beta is None: 
        raise ValueError('alpha, or beta, or both are required')

    if beta is None:
        beta = np.sqrt(1.0 - np.conj(alpha) * alpha)

    if alpha is None:
        alpha = np.sqrt(1.0 - np.conj(beta) * beta)

    if not math.isclose(np.conj(alpha) * alpha + 
                        np.conj(beta) * beta, 1.0): 
        raise ValueError( 'Qubit probabilities do not add up to 1')

    qb = np.zeros(2, dtype=tensor.tensor_type())
    qb[0] = alpha 
    qb[1] = beta
    return State(qb)

