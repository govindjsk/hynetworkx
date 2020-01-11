import numpy as np

def get_entropy(p):
    p = np.array(p)
    p = p[p>0]
    if p.size == 0:
        return 0
    p = p/sum(p)
    return -sum(p*np.log2(p))



def get_cond_entropy(pq_counts, q):
    q = np.array(q)
    sum_q = np.sum(q)
    if sum_q == 0:
        return 0
    else:
        q = q / sum_q
    p_given_q = np.array(pq_counts)
    p_given_q = p_given_q / np.sum(p_given_q, axis=0)
#     get_entropy_array = np.vectorize(get_entropy)
    realized_cond_entropies = np.apply_along_axis(get_entropy, 0, p_given_q)
    entropy = np.sum(q*realized_cond_entropies)
    return entropy




def get_mutual_info(pq_counts):
    pq_counts = np.array(pq_counts)
    p_counts = np.sum(pq_counts, axis=1)
    q_counts = np.sum(pq_counts, axis=0)
    p = p_counts / np.sum(p_counts)
    q = q_counts / np.sum(q_counts)
    entropy = get_entropy(p)
    cond_entropy = get_cond_entropy(pq_counts, q)
    mutual_info = entropy - cond_entropy
    return mutual_info



def test():
    # Entropy Tests
    assert get_entropy([0]) == 0
    assert get_entropy([1]) == 0
    assert get_entropy([]) == 0
    assert get_entropy([0, 1]) == 0
    assert get_entropy([1, 1]) == 1
    assert round(get_entropy([1, 3]), 6) == 0.811278
    
    # Conditional Entropy Tests
    assert round(get_cond_entropy([[3, 2], [5, 4]], [1, 3]), 6) == 0.927330
    assert get_cond_entropy([[]], []) == 0
    assert get_cond_entropy([[0]], [0]) == 0
    assert get_cond_entropy([[1]], [1]) == 0
    assert get_cond_entropy([[0, 1]], [0, 1]) == 0
    assert get_cond_entropy([[1, 1]], [1, 1]) == 0
    assert round(get_cond_entropy([[1], [3]], [1]), 6) == 0.811278
    assert round(get_cond_entropy([[1, 1], [3, 3]], [3, 1]), 6) == 0.811278
    
    # Mutual Information Tests
    assert round(get_mutual_info([[12, 45], [0, 45]]), 6) == 0.10764
    assert round(get_mutual_info([[22, 25, 2, 8], [0, 8, 0, 37]]), 6) == 0.433599
    

if __name__ == '__main__':
    test()
