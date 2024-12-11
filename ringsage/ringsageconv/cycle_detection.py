import networkx as nx

from torch_geometric.utils import to_networkx


def is_minimal_cycle(G, cycle):
    n = len(cycle)
    for i in range(n):
        for j in range(i + 2, n):
            if j - i < n - 1:
                if G.has_edge(cycle[i], cycle[j]):
                    return False
    return True


def get_minimal_cycles(G):
    all_cycles = list(nx.simple_cycles(G))
    all_cycles.sort(key=len)

    minimal_cycles = []
    for cycle in all_cycles:
        if is_minimal_cycle(G, cycle):
            minimal_cycles.append(cycle)

    return minimal_cycles


def get_cycle_info(pyg_graph):
    # Convert PyG graph to NetworkX
    nx_graph = to_networkx(pyg_graph, to_undirected=True)

    # Find minimal cycles
    cycles = get_minimal_cycles(nx_graph)

    # Create cycle_info dictionary
    cycle_info = {}

    for i, cycle in enumerate(cycles):
        cycle_key = i
        cycle_info[cycle_key] = list(cycle)

    return cycle_info
