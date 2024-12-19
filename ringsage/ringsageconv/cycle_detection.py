import networkx as nx

from torch_geometric.utils import to_networkx


def is_minimal_cycle(G, cycle):
    """
    Determines whether a given cycle in the graph is minimal by checking for internal connections.

    Args:
        G (networkx.Graph): The input graph.
        cycle (list): A list of nodes representing a cycle.

    Returns:
        bool: True if the cycle is minimal, False otherwise.
    """

    # Check for internal connections
    n = len(cycle)

    for i in range(n):
        for j in range(i + 2, n):
            if j - i < n - 1:
                if G.has_edge(cycle[i], cycle[j]):
                    return False
    return True

def get_minimal_cycles(G):
    """
    Extracts minimal cycles from a graph.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        list: A list of minimal cycles, where each cycle is represented as a list of nodes.
    """

    # Find all cycles in the graph
    all_cycles = list(nx.simple_cycles(G))

    # Sort cycles based on number of nodes
    all_cycles.sort(key=len)

    minimal_cycles = []
    for cycle in all_cycles:
        # Check if cycle is minimal
        if is_minimal_cycle(G, cycle):
            # If the cycle is minimal, add it to the list of minimal cycles
            minimal_cycles.append(cycle)

    return minimal_cycles


def get_cycle_info(pyg_graph):
    """
    Converts a PyTorch-Geometric (PyG) graph to a NetworkX graph and extracts minimal cycles.

    Args:
        pyg_graph (torch_geometric.data.Data): The input PyG graph.

    Returns:
        dict: A dictionary where keys are cycle indices and values are lists of nodes in each cycle.
    """

    # Convert PyG graph to NetworkX
    nx_graph = to_networkx(pyg_graph, to_undirected=True)

    # Find minimal cycles
    cycles = get_minimal_cycles(nx_graph)

    # Create cycle_info dictionary
    cycle_info = {i: list(cycle) for i, cycle in enumerate(cycles)}

    return cycle_info
