def select_peers(peers, selector, n: int = 0, frac: int = 0):
    selected_peers = []
    if peers == None or len(peers) == 0:
        print('Error: peer list is empty')
    else:
        if n >= 0 and frac == 0:
            selected_peers = selector.select_n_random_peers(peers, n)
        if n == 0 and frac >= 0:
            selected_peers = selector.select_frac_random_peers(peers, frac)

    return selected_peers