import grinpy as gp
import networkx as nx
import numpy as np
import itertools

__all__ = ["compute", "factors_compute"]


def compute(G, property):
    """
    Returns the value of a given graph property for a given graph G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    property : string
        The name of the graph property to be calculated for the graph G.

    Returns
    -------
    int or float or bool
        The value of the graph property for the graph G.
    """
    if property == "k_slater_index":
        return k_slater_index(G)
    elif property == "wiener_index":
        return gp.wiener_index(G)
    elif property == "vertex_cover_number":
        return vertex_cover_number(G)
    elif property == "k_residual_index":
        return k_residual_index(G)
    elif property == "order":
        return gp.number_of_nodes(G)
    elif property == "size":
        return gp.number_of_edges(G)
    elif property == "edge_domination_number":
        LG = gp.line_graph(G)
        return gp.domination_number(LG)
    elif property == "min_maximal_matching_number":
        LG = gp.line_graph(G)
        return gp.independent_domination_number(LG)
    elif property == "roman_domination_number":
        return roman_domination_number(G)
    elif property == "double_roman_domination_number":
        return double_roman_domination_number(G)
    elif property == "two_rainbow_domination_number":
        return two_rainbow_domination_number(G)
    elif property == "three_rainbow_domination_number":
        return three_rainbow_domination_number(G)
    elif property == "restrained_domination_number":
        return restrained_domination_number(G)
    elif property == "(order - domination_number)":
        return gp.number_of_nodes(G) - gp.domination_number(G)
    elif property == "(order - total_domination_number)":
        return gp.number_of_nodes(G) - gp.total_domination_number(G)
    elif property == "(order - connected_domination_number)":
        return gp.number_of_nodes(G) - gp.connected_domination_number(G)
    elif property == "(order - independence_number)":
        return gp.number_of_nodes(G) - gp.independence_number(G)
    elif property == "(order - power_domination_number)":
        return gp.number_of_nodes(G) - gp.power_domination_number(G)
    elif property == "(order - zero_forcing_number)":
        return gp.number_of_nodes(G) - gp.zero_forcing_number(G)
    elif property == "(order - diameter)":
        return gp.number_of_nodes(G) - gp.diameter(G)
    elif property == "(order - radius)":
        return gp.number_of_nodes(G) - gp.radius(G)
    elif property == "(order - triameter)":
        return gp.number_of_nodes(G) - gp.triameter(G)
    elif property == "(size - diameter)":
        return gp.number_of_edges(G) - gp.diameter(G)
    elif property == "(size - radius)":
        return gp.number_of_edges(G) - gp.radius(G)
    elif property == "(size - triameter)":
        return gp.number_of_edges(G) - gp.triameter(G)
    elif property == "(order - independent_domination_number)":
        return gp.number_of_nodes(G) - gp.independent_domination_number(G)
    elif property == "(order - chromatic_number)":
        return gp.number_of_nodes(G) - gp.chromatic_number(G)
    elif property == "(order - matching_number)":
        return gp.number_of_nodes(G) - gp.matching_number(G)
    elif property == "(order - min_maximal_matching_number)":
        return gp.number_of_nodes(G) - gp.min_maximal_matching_number(G)
    elif property == "(size - matching_number)":
        return gp.number_of_edges(G) - gp.matching_number(G)
    elif property == "(size - min_maximal_matching_number)":
        return gp.number_of_edges(G) - gp.min_maximal_matching_number(G)
    elif property == "(order - min_degree)":
        return gp.number_of_nodes(G) - gp.min_degree(G)
    elif property == "(order - max_degree)":
        return gp.number_of_nodes(G) - gp.max_degree(G)
    elif property == "(order - clique_number)":
        return gp.number_of_nodes(G) - gp.clique_number(G)
    elif property == "(order - residue)":
        return gp.number_of_nodes(G) - gp.residue(G)
    elif property == "(order - annihilation_number)":
        return gp.number_of_nodes(G) - gp.annihilation_number(G)
    elif property == "(order - sub_total_domination_number)":
        return gp.number_of_nodes(G) - gp.sub_total_domination_number(G)
    elif property == "(order - slater)":
        return gp.number_of_nodes(G) - gp.slater(G)
    elif property == "(order - k_slater_index)":
        return gp.number_of_nodes(G) - k_slater_index(G)
    elif property == "(order - k_residual_index)":
        return gp.number_of_nodes(G) - k_residual_index(G)
    elif property == "min_edge_cover":
        return len(gp.min_edge_cover(G))
    elif property == "[(annihilation_number + residue)/ max_degree]":
        return (gp.annihilation_number(G) + gp.residue(G)) / gp.max_degree(G)
    elif property == "[order/ max_degree]":
        return gp.number_of_nodes(G) / gp.max_degree(G)
    elif property == "[order/ (max_degree + 1)]":
        return gp.number_of_nodes(G) / (gp.max_degree(G) + 1)
    elif property == "[order/ (max_degree - 1)]":
        return gp.number_of_nodes(G) / (gp.max_degree(G) - 1)
    elif property == "[order/ (max_degree + 2)]":
        return gp.number_of_nodes(G) / (gp.max_degree(G) + 2)
    elif property == "(residue + annihilation_number)":
        return gp.residue(G) + gp.annihilation_number(G)
    elif property == "graph_energy":
        return graph_energy(G)
    elif property == "square_positive_energy":
        return square_positive_energy(G)
    elif property == "square_negative_energy":
        return square_negative_energy(G)
    elif property == "positive_semidefinite_zero_forcing_number":
        return positive_semidefinite_zero_forcing_number(G)
    elif property == "second_largest_eigenvalue":
        return second_largest_eigenvalue(G)
    elif property == "a connected graph":
        return gp.is_connected(G)
    elif property == "a connected and planar graph":
        return gp.is_connected(G) and gp.check_planarity(G)[0]
    elif property == "a connected and regular graph":
        return gp.is_connected(G) and gp.min_degree(G) == gp.max_degree(G)
    elif property == "a connected and cubic graph":
        return gp.is_connected(G) and gp.min_degree(G) == 3 and gp.max_degree(G) == 3
    elif property == "a connected graph which is not K_n":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False
    elif property == "a connected and triangle-free graph":
        return gp.is_connected(G) and set(gp.triangles(G).values()) == {0}
    elif property == "a connected and claw-free graph":
        return gp.is_connected(G) and gp.is_claw_free(G)
    elif property == "a connected and chordal graph":
        return gp.is_connected(G) and gp.is_chordal(G)
    elif property == "a tree graph":
        return gp.is_connected(G) and gp.is_tree(G)
    elif property == "a connected and at-free graph":
        return gp.is_connected(G) and gp.is_at_free(G)
    elif property == "an eulerian graph":
        return gp.is_connected(G) and gp.is_eulerian(G)
    elif property == "a connected and bipartite graph":
        return gp.is_connected(G) and gp.is_bipartite(G)
    elif property == "a connected graph with maximum degree at most 3":
        return gp.is_connected(G) and gp.max_degree(G) <= 3
    elif property == "a connected graph which is not K_n and has maximum degree at most 3":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False and gp.max_degree(G) <= 3
    elif property == "a connected and cubic graph which is not K_4":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False and gp.min_degree(G) == 3 and gp.max_degree(G) == 3
    elif property == "a connected, claw-free, and cubic graph":
        return gp.is_connected(G) and gp.min_degree(G) == 3 and gp.max_degree(G) == 3 and gp.is_claw_free(G)
    elif property == "a connected, planar, and cubic graph":
        return gp.is_connected(G) and gp.min_degree(G) == 3 and gp.max_degree(G) == 3 and gp.is_planar(G)
    elif property == "a connected graph with a total domination number equal to the domination number":
        return gp.is_connected(G) and gp.total_domination_number(G) == gp.domination_number(G)
    elif property == "a connected and well-covered graph":
        return gp.is_connected(G) and gp.independence_number(G) == gp.independent_domination_number(G)
    elif property == "a connected and Class-1 graph":
        new_G = gp.line_graph(G)
        return gp.is_connected(G) and gp.chromatic_number(new_G) == gp.max_degree(G)
    elif property == "a connected and Class-2 graph":
        new_G = gp.line_graph(G)
        return gp.is_connected(G) and gp.chromatic_number(new_G) == gp.max_degree(G) + 1
    elif property == "a connected graph with diameter at most 3":
        return gp.is_connected(G) and gp.diameter(G) <= 3
    elif property == "a connected and planar graph with diameter at most 3":
        return gp.is_connected(G) and gp.is_planar(G) and gp.diameter(G) <= 3
    elif property == "a connected_graph with min_degree at least 2":
        return gp.is_connected(G) and gp.min_degree(G) >= 2
    elif property == "a connected_graph with min_degree at least 3":
        return gp.is_connected(G) and gp.min_degree(G) >= 3
    elif property == "a connected_graph with min_degree at least 4":
        return gp.is_connected(G) and gp.min_degree(G) >= 4
    elif property == "a connected and bipartite graph with min_degree at least 2":
        return gp.is_connected(G) and gp.is_bipartite(G) and gp.min_degree(G) >= 2
    elif property == "a connected and bipartite graph with min_degree at least 3":
        return gp.is_connected(G) and gp.is_bipartite(G) and gp.min_degree(G) >= 3
    elif property == "a connected and bipartite graph with min_degree at least 4":
        return gp.is_connected(G) and gp.is_bipartite(G) and gp.min_degree(G) >= 4
    elif property == "a connected and planar graph with min_degree at least 2":
        return gp.is_connected(G) and gp.is_planar(G) and gp.min_degree(G) >= 2
    elif property == "a connected and planar graph with min_degree at least 3":
        return gp.is_connected(G) and gp.is_planar(G) and gp.min_degree(G) >= 3
    elif property == "a connected and planar graph with min_degree at least 4":
        return gp.is_connected(G) and gp.is_planar(G) and gp.min_degree(G) >= 4
    elif property == "a connected graph which is not K_n with min_degree at least 2":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False and gp.min_degree(G) >= 2
    elif property == "a connected graph which is not K_n with min_degree at least 3":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False and gp.min_degree(G) >= 3
    elif property == "a connected graph which is not K_n with min_degree at least 4":
        return gp.is_connected(G) and gp.is_isomorphic(G, gp.complete_graph(gp.number_of_nodes(G))) == False and gp.min_degree(G) >= 4
    elif property == "a connected and triangle-free graph with min_degree at least 2":
        return gp.is_connected(G) and set(gp.triangles(G).values()) == {0} and gp.min_degree(G) >= 2
    elif property == "a connected and triangle-free graph with min_degree at least 3":
        return gp.is_connected(G) and set(gp.triangles(G).values()) == {0} and gp.min_degree(G) >= 3
    elif property == "a connected and triangle-free graph with min_degree at least 4":
        return gp.is_connected(G) and set(gp.triangles(G).values()) == {0} and gp.min_degree(G) >= 4
    elif property == "a connected and at-free graph with min_degree at least 2":
        return gp.is_connected(G) and gp.is_at_free(G) and gp.min_degree(G) >= 2
    elif property == "a connected and at-free graph with min_degree at least 3":
        return gp.is_connected(G) and gp.is_at_free(G) and gp.min_degree(G) >= 3
    elif property == "a connected and at-free graph with min_degree at least 4":
        return gp.is_connected(G) and gp.is_at_free(G) and gp.min_degree(G) >= 4
    elif property == "a connected and claw-free graph with min_degree at least 2":
        return gp.is_connected(G) and gp.is_claw_free(G) and gp.min_degree(G) >= 2
    elif property == "a connected and claw-free graph with min_degree at least 3":
        return gp.is_connected(G) and gp.is_claw_free(G) and gp.min_degree(G) >= 3
    elif property == "a connected and claw-free graph with min_degree at least 4":
        return gp.is_connected(G) and gp.is_claw_free(G) and gp.min_degree(G) >= 4
    elif property == "a connected and chordal graph with min_degree at least 2":
        return gp.is_connected(G) and gp.is_chordal(G) and gp.min_degree(G) >= 2
    elif property == "a connected and chordal graph with min_degree at least 3":
        return gp.is_connected(G) and gp.is_chordal(G) and gp.min_degree(G) >= 3
    elif  property == "a connected graph with min_degree at least 2 and maximum degree at most 3":
        return gp.is_connected(G) and gp.min_degree(G) >= 2 and gp.max_degree(G) <= 3
    elif property == "a connected and diamond-free graph":
        return gp.is_connected(G) and is_diamond_free(G)
    elif property == "a connected, cubic, and diamond-free graph":
        return gp.is_connected(G) and is_cubic_and_diamond_free(G)
    elif property == "a connected and bull-free graph":
        return gp.is_connected(G) and gp.is_bull_free(G)
    elif property == "semitotal_domination_number":
        return semitotal_domination_number(G)
    elif property == "a block graph":
        return is_block_graph(G)
    elif property == "outer_connected_domination_number":
        return outer_connected_domination_number(G)
    elif property == "strong_harmonic_index":
        return strong_harmonic_index(G)
    elif property == "reciprocal_first_zagreb_index":
        return reciprocal_first_zagreb_index(G)
    elif property == "reciprocal_second_zagreb_index":
        return reciprocal_second_zagreb_index(G)
    elif property == "reciprocal_harary_index":
        return reciprocal_harary_index(G)
    elif property == "reciprocal_second_zagreb_variation":
        return reciprocal_second_zagreb_variation(G)
    elif property == "reciprocal_randic_index":
        return reciprocal_randic_index(G)
    elif property == "reciprocal_augmented_zagreb_index":
        return reciprocal_augmented_zagreb_index(G)
    elif property == "reciprocal_sum_connectivity_index":
        return reciprocal_sum_connectivity_index(G)
    elif property == "reciprocal_hyper_zagreb_index":
        return reciprocal_hyper_zagreb_index(G)
    elif property == "reciprocal_geometric_arithmetic_index":
        return reciprocal_geometric_arithmetic_index(G)
    elif property == "inverse_degree_plus_one_sum":
        return inverse_degree_plus_one_sum(G)
    elif property == "inverse_degree_plus_two_sum":
        return inverse_degree_plus_two_sum(G)
    elif property == "inverse_edge_degree_plus_one_sum":
        return inverse_edge_degree_plus_one_sum(G)
    elif property == "inverse_edge_degree_plus_two_sum":
        return inverse_edge_degree_plus_two_sum(G)


    # New 2-degree based indices
    elif property == "first_zagreb_index_2_degree":
        return first_zagreb_index_2_degree(G)
    elif property == "second_zagreb_index_2_degree":
        return second_zagreb_index_2_degree(G)
    elif property == "reciprocal_first_zagreb_index_2_degree":
        return reciprocal_first_zagreb_index_2_degree(G)
    elif property == "reciprocal_second_zagreb_index_2_degree":
        return reciprocal_second_zagreb_index_2_degree(G)
    elif property == "average_degree_2_degree":
        return average_degree_2_degree(G)
    elif property == "reciprocal_randic_index_2_degree":
        return reciprocal_randic_index_2_degree(G)
    elif property == "reciprocal_sum_connectivity_index_2_degree":
        return reciprocal_sum_connectivity_index_2_degree(G)
    elif property == "hyper_zagreb_index_2_degree":
        return hyper_zagreb_index_2_degree(G)
    elif property == "reciprocal_geometric_arithmetic_index_2_degree":
        return reciprocal_geometric_arithmetic_index_2_degree(G)

    elif property == "augmented_average_edge_degree":
        return augmented_average_edge_degree(G)

    elif property == "a connected graph that is a line graph":
        return is_line_graph_modified(G)

    else:
        return getattr(gp, property)(G)

def factors_compute(G, H, property):
    # Split the property string by the operation (either + or *)
    if '+' in property:
        operation = '+'
    elif '*' in property:
        operation = '*'
    else:
        raise ValueError("Property must contain either '*' for multiplication or '+' for addition.")

    # Extract the properties for G and H
    property_G, property_H = property.split(f" {operation} ")


    # Compute the values for G and H using the respective properties
    value_G = compute(G, property_G)
    value_H = compute(H, property_H)

    # Apply the operation
    if operation == '+':
        return value_G + value_H
    elif operation == '*':
        return value_G * value_H



def k_slater_index(G):
    """Return a the smallest integer k so that the sub-k-domination number
    of G is at least the domination number of G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    number
        The smallest ineteger k such that gp.domination_number(G) <= gp.sub_k_domination_number(G, k).
    """
    k = 1
    while gp.sub_k_domination_number(G, k) < gp.domination_number(G):
        k += 1
    return k

def vertex_cover_number(G):
    """Return a the size of smallest vertex cover in the graph G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    number
        The size of a smallest vertex cover of G.
    """
    return gp.number_of_nodes(G) - gp.independence_number(G)

def k_residual_index(G):
    """Return a the smallest integer k so that the k-residue of G is at least the
    independence number of G.
    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.
    Returns
    -------
    number
        The smallest ineteger k such that gp.independence_number(G) <= gp.k_residue(G, k).

    """
    k = 1
    while gp.k_residue(G, k) < gp.independence_number(G):
        k += 1
    return k

def is_diamond_free(G):
    """Return True if the graph G is diamond-free, and False otherwise.

    A graph is diamond-free if it does not contain a diamond as an induced subgraph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    bool
        True if G is diamond-free, and False otherwise.
    """
    diamond = gp.complete_graph(4)
    diamond.remove_edge(1, 2)
     # enumerate over all possible combinations of 5 vertices contained in G
    for S in set(itertools.combinations(G.nodes(), 4)):
        H = G.subgraph(list(S))
        if gp.is_isomorphic(H, diamond):
            return False
    # if the above loop completes, the graph is bull-free
    return True

def is_cubic_and_diamond_free(G):
    """Return True if the graph G is cubic and diamond-free, and False otherwise.

    A graph is cubic if every vertex has degree 3, and diamond-free if it does not contain a diamond as an induced subgraph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    bool
        True if G is cubic and diamond-free, and False otherwise.
    """
    return gp.min_degree(G) == gp.max_degree(G) == 3 and is_diamond_free(G)

import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

def closed_neighborhood(G, node):
    """Return the closed neighborhood of a node."""
    return set(G.neighbors(node)).union({node})

def min_semitotal_dominating_set_ilp(G):
    """Return a smallest semitotal dominating set in the graph.

    A semitotal dominating set in a graph *G* is a set *D* of nodes of *G* such that:
    1. *D* is a dominating set.
    2. Each vertex in *D* has another vertex in *D* within distance 2.

    This method uses integer programming to compute a smallest semitotal dominating set.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    set
        A set of nodes in a smallest semitotal dominating set in the graph.
    """
    prob = LpProblem("min_semitotal_dominating_set", LpMinimize)
    variables = {node: LpVariable("x{}".format(i + 1), 0, 1, LpBinary) for i, node in enumerate(G.nodes())}

    # Set the semitotal domination number objective function
    prob += lpSum([variables[n] for n in variables])

    # Constraint 1: Dominating set constraint
    for node in G.nodes():
        combination = [variables[n] for n in variables if n in closed_neighborhood(G, node)]
        prob += lpSum(combination) >= 1

    # Constraint 2: Semitotal domination constraint
    for node in G.nodes():
        if len(closed_neighborhood(G, node).intersection(G.nodes())) > 1:
            prob += variables[node] + lpSum(variables[neighbor] for neighbor in G.neighbors(node) if neighbor != node) >= 2

    prob.solve()
    solution_set = {node for node in variables if variables[node].value() == 1}
    return solution_set

def semitotal_domination_number(G):
    """Return the semitotal domination number of the graph.

    The semitotal domination number of a graph *G* is the size of a smallest semitotal dominating set in *G*.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The semitotal domination number of the graph.
    """
    return len(min_semitotal_dominating_set_ilp(G))

def graph_energy(G):
    """
    Compute the energy of a graph G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The energy of the graph G.
    """
    # Step 1: Compute the adjacency matrix A(G)
    A = nx.adjacency_matrix(G).todense()

    # Step 2: Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(A)

    # Step 3: Calculate the energy as the sum of the absolute values of the eigenvalues
    energy = sum(np.abs(eigenvalues))

    return round(energy)

def square_positive_energy(G):
    """
    Compute the square positive energy of a graph G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The square positive energy of the graph G.
    """
    # Step 1: Compute the adjacency matrix A(G)
    A = nx.adjacency_matrix(G).todense()

    # Step 2: Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(A)

    positive_eigenvalues_sqaures = [eig**2 for eig in eigenvalues if eig > 0]
    # negative_eigenvalues = [eig for eig in eigenvalues if eig < 0]

    # Step 3: Calculate the energy as the sum of the absolute values of the eigenvalues
    energy = sum(positive_eigenvalues_sqaures)

    return round(energy)

def square_negative_energy(G):
    """
    Compute the square negative energy of a graph G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The square negative energy of the graph G.
    """
    # Step 1: Compute the adjacency matrix A(G)
    A = nx.adjacency_matrix(G).todense()

    # Step 2: Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(A)

    negative_eigenvalues_sqaures = [eig**2 for eig in eigenvalues if eig < 0]

    # Step 3: Calculate the energy as the sum of the absolute values of the eigenvalues
    energy = sum(negative_eigenvalues_sqaures)

    return round(energy)


from itertools import combinations
from networkx import is_connected, Graph, neighbors

def is_psd_forcing_vertex(G, v, black_set, component):
    """
    Return whether or not v can force any white vertex in the component.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    v : node
        A single node in G.

    black_set : set
        A set of black vertices in G.

    component : set
        A set of vertices in a component of G - black_set.

    Returns
    -------
    tuple
        A tuple (True, w) if v can force the white vertex w in the component, (False, None) otherwise.
    """
    set_neighbors = set(neighbors(G, v))
    white_neighbors_in_component = set_neighbors.intersection(component)

    if len(white_neighbors_in_component) == 1:
        w = white_neighbors_in_component.pop()
        return (True, w)
    return (False, None)


def psd_color_change(G, black_set):
    """
    Apply the PSD color change rule repeatedly until no more vertices can be forced.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    black_set : set
        A set of initial black vertices.

    Returns
    -------
    set
        The derived set of black vertices.
    """
    black_set = set(black_set)
    white_set = set(G.nodes()) - black_set

    while True:
        new_black = set()
        components = [set(c) for c in nx.connected_components(G.subgraph(white_set))]

        for component in components:
            for v in black_set:
                can_force, w = is_psd_forcing_vertex(G, v, black_set, component)
                if can_force:
                    new_black.add(w)

        if not new_black:
            break

        black_set.update(new_black)
        white_set -= new_black

    return black_set


def is_psd_zero_forcing_set(G, black_set):
    """
    Return whether or not the vertices in black_set comprise a PSD zero forcing set in G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    black_set : set
        A set of initial black vertices.

    Returns
    -------
    boolean
        True if the black_set is a PSD zero forcing set in G. False otherwise.
    """
    derived_set = psd_color_change(G, black_set)
    return len(derived_set) == G.order()


def min_psd_zero_forcing_set(G):
    """
    Return a smallest PSD zero forcing set in G.

    The method used to compute the set is brute force.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    list
        A list of nodes in a smallest PSD zero forcing set in G.
    """
    for i in range(1, G.order() + 1):
        for black_set in combinations(G.nodes(), i):
            if is_psd_zero_forcing_set(G, black_set):
                return list(black_set)


def positive_semidefinite_zero_forcing_number(G):
    """
    Return the PSD zero forcing number of G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    int
        The PSD zero forcing number of G.
    """
    return len(min_psd_zero_forcing_set(G))

import pulp
import networkx as nx

def semitotal_domination_number(G):
    # Map node labels to indices
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inv_node_mapping = {idx: node for node, idx in node_mapping.items()}
    n = len(G)

    # Create a linear-integer problem
    prob = pulp.LpProblem("SemitotalDomination", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", range(n), 0, 1, pulp.LpBinary)

    # Objective function
    prob += pulp.lpSum(x[i] for i in range(n)), "MinimizeSemitotalDominationSet"

    # Dominating set constraint
    for i in range(n):
        node = inv_node_mapping[i]
        prob += x[i] + pulp.lpSum(x[node_mapping[j]] for j in G.neighbors(node)) >= 1, f"DominatingSet_{i}"

    # Semitotal constraint
    for i in range(n):
        node = inv_node_mapping[i]
        distance_2_neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=2).keys())
        distance_2_neighbors.discard(node)  # Exclude the vertex itself
        prob += x[i] <= pulp.lpSum(x[node_mapping[j]] for j in distance_2_neighbors), f"Semitotal_{i}"

    # Solve the problem
    prob.solve()

    # Get the solution
    semitotal_dominating_set = [inv_node_mapping[i] for i in range(n) if x[i].varValue > 0.5]
    return len(semitotal_dominating_set), semitotal_dominating_set

def second_largest_eigenvalue(G):
    """
    Compute the second largest eigenvalue of the adjacency matrix of a graph G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    float
        The second largest eigenvalue of the adjacency matrix of G.
    """
    # Step 1: Compute the adjacency matrix A(G)
    A = nx.adjacency_matrix(G).todense()

    # Step 2: Compute the eigenvalues of the adjacency matrix
    eigenvalues = np.linalg.eigvals(A)

    # Step 3: Sort the eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    value = round(sorted_eigenvalues[1])

    value = int(value)

    # Step 4: Return the second largest eigenvalue
    return value

def is_complete_graph(G):
    """Check if a graph G is a complete graph."""
    n = G.number_of_nodes()
    # A complete graph with n nodes has n*(n-1)/2 edges
    return G.number_of_edges() == n * (n - 1) // 2

def is_block_graph(G):
    # Check if the graph is connected
    if not nx.is_connected(G):
        return False

    # Find all biconnected components (blocks)
    blocks = list(nx.biconnected_components(G))

    for block in blocks:
        subgraph = G.subgraph(block)
        # Check if the subgraph (block) is a clique
        if not is_complete_graph(subgraph):
            return False

    return True

import itertools

def is_dominating_set(G, S):
    X = G.nodes() - S
    for u in X:
        if not any(v in S for v in G.neighbors(u)):
            return False
    return True

def complement_is_connected(G, S):
    X = G.nodes() - S
    return nx.is_connected(G.subgraph(X))

def is_outer_connected_dominating_set(G, S):
    return is_dominating_set(G, S) and complement_is_connected(G, S)

def min_outer_connected_dominating_set(G):
    n = len(G.nodes())
    min_set = None

    for r in range(1, n + 1):  # Try all subset sizes
        for S in itertools.combinations(G.nodes(), r):
            S = set(S)
            if is_outer_connected_dominating_set(G, S):
                return S

def outer_connected_domination_number(G):
    return len(min_outer_connected_dominating_set(G))


from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value

def roman_domination(graph):
    # Initialize the problem
    prob = LpProblem("RomanDomination", LpMinimize)

    # Define variables x_v, y_v for each vertex v
    x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in graph.nodes()}
    y = {v: LpVariable(f"y_{v}", cat=LpBinary) for v in graph.nodes()}

    # Objective function: min sum(x_v + 2*y_v)
    prob += lpSum(x[v] + 2 * y[v] for v in graph.nodes()), "MinimizeCost"

    # Dominance Constraint: x_v + y_v + sum(y_u for u in N(v)) >= 1 for all v
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += x[v] + y[v] + lpSum(y[u] for u in neighbors) >= 1, f"DominanceConstraint_{v}"

    # Mutual Exclusivity: x_v + y_v <= 1 for all v
    for v in graph.nodes():
        prob += x[v] + y[v] <= 1, f"ExclusivityConstraint_{v}"

    # Solve the problem
    prob.solve()

    # Extract solution
    solution = {
        "x": {v: value(x[v]) for v in graph.nodes()},
        "y": {v: value(y[v]) for v in graph.nodes()},
        "objective": value(prob.objective)
    }

    return solution

def roman_domination_number(graph):
    solution = roman_domination(graph)
    return solution["objective"]


def double_roman_domination(graph):
    # Initialize the problem
    prob = LpProblem("DoubleRomanDomination", LpMinimize)

    # Define variables x_v, y_v, z_v for each vertex v
    x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in graph.nodes()}
    y = {v: LpVariable(f"y_{v}", cat=LpBinary) for v in graph.nodes()}
    z = {v: LpVariable(f"z_{v}", cat=LpBinary) for v in graph.nodes()}

    # Objective function: min sum(x_v + 2*y_v + 3*z_v)
    prob += lpSum(x[v] + 2 * y[v] + 3 * z[v] for v in graph.nodes()), "MinimizeCost"

    # Constraint (1b): xv + yv + zv + 1/2 * sum(yu for u in N(v)) + sum(zu for u in N(v)) >= 1
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += x[v] + y[v] + z[v] + 0.5 * lpSum(y[u] for u in neighbors) + lpSum(z[u] for u in neighbors) >= 1, f"Constraint_1b_{v}"

    # Constraint (1c): sum(yu + zu) >= xv for each vertex v
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        prob += lpSum(y[u] + z[u] for u in neighbors) >= x[v], f"Constraint_1c_{v}"

    # Constraint (1d): xv + yv + zv <= 1
    for v in graph.nodes():
        prob += x[v] + y[v] + z[v] <= 1, f"Constraint_1d_{v}"

    # Solve the problem
    prob.solve()

    # Extract solution
    solution = {
        "x": {v: value(x[v]) for v in graph.nodes()},
        "y": {v: value(y[v]) for v in graph.nodes()},
        "z": {v: value(z[v]) for v in graph.nodes()},
        "objective": value(prob.objective)
    }

    return solution

def double_roman_domination_number(graph):
    solution = double_roman_domination(graph)
    return solution["objective"]


def solve_rainbow_domination(G, k):
    # Create a PuLP problem instance
    prob = pulp.LpProblem("Rainbow_Domination", pulp.LpMinimize)

    # Create binary variables f_vi where f_vi = 1 if vertex v is colored with color i
    f = pulp.LpVariable.dicts("f", ((v, i) for v in G.nodes for i in range(1, k+1)), cat='Binary')

    # Create binary variables x_v where x_v = 1 if vertex v is uncolored
    x = pulp.LpVariable.dicts("x", G.nodes, cat='Binary')

    # Objective function: Minimize the total number of colored vertices
    prob += pulp.lpSum(f[v, i] for v in G.nodes for i in range(1, k+1)), "Minimize total colored vertices"

    # Constraint 1: Each vertex is either colored with one of the k colors or remains uncolored
    for v in G.nodes:
        prob += pulp.lpSum(f[v, i] for i in range(1, k+1)) + x[v] == 1, f"Color or Uncolored constraint for vertex {v}"

    # Constraint 2: If a vertex is uncolored (x_v = 1), it must be adjacent to vertices colored with all k colors
    for v in G.nodes:
        for i in range(1, k+1):
            # Ensure that uncolored vertex v is adjacent to a vertex colored with color i
            prob += pulp.lpSum(f[u, i] for u in G.neighbors(v)) >= x[v], f"Rainbow domination for vertex {v} color {i}"

    # Solve the problem using PuLP's default solver
    prob.solve()

    # Output results
    # print("Status:", pulp.LpStatus[prob.status])

    # Print which vertices are colored and with what color
    colored_vertices = [(v, i) for v in G.nodes for i in range(1, k+1) if pulp.value(f[v, i]) == 1]
    uncolored_vertices = [v for v in G.nodes if pulp.value(x[v]) == 1]

    print(f"Colored vertices: {colored_vertices}")
    print(f"Uncolored vertices: {uncolored_vertices}")

    return colored_vertices, uncolored_vertices


def rainbow_domination_number(G, k):
    colored_vertices, uncolored_vertices = solve_rainbow_domination(G, k)
    return len(colored_vertices)


def two_rainbow_domination_number(G):
    return rainbow_domination_number(G, 2)

def three_rainbow_domination_number(G):
    return rainbow_domination_number(G, 3)


def minimum_restrained_dominating_set(G):
    # Initialize the linear programming problem
    prob = pulp.LpProblem("MinimumRestrainedDomination", pulp.LpMinimize)

    # Decision variables: x_v is 1 if vertex v is in the restrained dominating set, 0 otherwise
    x = {v: pulp.LpVariable(f"x_{v}", cat="Binary") for v in G.nodes()}

    # Objective: Minimize the sum of x_v
    prob += pulp.lpSum(x[v] for v in G.nodes()), "Objective"

    # Constraint 1: Domination condition
    for v in G.nodes():
        prob += x[v] + pulp.lpSum(x[u] for u in G.neighbors(v)) >= 1, f"Domination_{v}"

    # Constraint 2: No isolated vertices in the complement of the dominating set
    for v in G.nodes():
        prob += pulp.lpSum(1 - x[u] for u in G.neighbors(v)) >= (1 - x[v]), f"NoIsolated_{v}"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract the solution
    restrained_dom_set = [v for v in G.nodes() if pulp.value(x[v]) == 1]

    return restrained_dom_set

def restrained_domination_number(G):
    restrained_dom_set = minimum_restrained_dominating_set(G)
    return len(restrained_dom_set)

def min_reciprocal_degree(G, v, w):
    return min(1/G.degree(v), 1/G.degree(w))

def strong_harmonic_index(G):
    return sum(min_reciprocal_degree(G, v, w) for v, w in G.edges())

# Reciprocal First Zagreb Index
def reciprocal_first_zagreb_index(G):
    return sum(1 / (d ** 2) for v, d in G.degree() if d > 0)

# Reciprocal Second Zagreb Index
def reciprocal_second_zagreb_index(G):
    return sum(1 / (G.degree(u) * G.degree(v)) for u, v in G.edges() if G.degree(u) > 0 and G.degree(v) > 0)

# # Reciprocal Estrada Index
# def reciprocal_estrada_index(G):
#     eigenvalues = np.linalg.eigvals(nx.adjacency_matrix(G).todense())
#     return sum(1 / np.exp(l) for l in eigenvalues if np.exp(l) > 0)

# Reciprocal Harary Index
def reciprocal_harary_index(G):
    return sum(1 / (nx.shortest_path_length(G, u, v) ** 2) for u in G for v in G if u != v)

# Reciprocal Second Zagreb Index Variation
def reciprocal_second_zagreb_variation(G):
    return sum(1 / (G.degree(u) + G.degree(v)) for u, v in G.edges() if (G.degree(u) + G.degree(v)) > 0)

# Reciprocal Randić Index
def reciprocal_randic_index(G):
    return sum(1 / np.sqrt(G.degree(u) * G.degree(v)) for u, v in G.edges() if G.degree(u) > 0 and G.degree(v) > 0)

# Reciprocal Augmented Zagreb Index
def reciprocal_augmented_zagreb_index(G):
    return sum(((G.degree(u) + G.degree(v) - 2) / (G.degree(u) * G.degree(v))) ** 3 for u, v in G.edges() if G.degree(u) > 0 and G.degree(v) > 0)

# Reciprocal Sum-connectivity Index
def reciprocal_sum_connectivity_index(G):
    return sum(1 / np.sqrt(G.degree(u) + G.degree(v)) for u, v in G.edges() if (G.degree(u) + G.degree(v)) > 0)

# Reciprocal Hyper-Zagreb Index
def reciprocal_hyper_zagreb_index(G):
    return sum(1 / (d * (d - 1)) for v, d in G.degree() if d > 1)

# Reciprocal Geometric-Arithmetic Index
def reciprocal_geometric_arithmetic_index(G):
    return sum(2 * np.sqrt(G.degree(u) * G.degree(v)) / (G.degree(u) + G.degree(v)) for u, v in G.edges() if (G.degree(u) + G.degree(v)) > 0)

def two_degree(G, v):
    return len(set(nx.single_source_shortest_path_length(G, v, cutoff=2)) - {v})

# 2-degree First Zagreb Index
def first_zagreb_index_2_degree(G):
    return sum(two_degree(G, v) ** 2 for v in G.nodes())

# 2-degree Second Zagreb Index
def second_zagreb_index_2_degree(G):
    return sum(two_degree(G, u) * two_degree(G, v) for u, v in G.edges())

# Reciprocal 2-degree First Zagreb Index
def reciprocal_first_zagreb_index_2_degree(G):
    return sum(1 / (two_degree(G, v) ** 2) for v in G.nodes() if two_degree(G, v) > 0)

# Reciprocal 2-degree Second Zagreb Index
def reciprocal_second_zagreb_index_2_degree(G):
    return sum(1 / (two_degree(G, u) * two_degree(G, v)) for u, v in G.edges() if two_degree(G, u) > 0 and two_degree(G, v) > 0)

# 2-degree Average Degree
def average_degree_2_degree(G):
    return sum(two_degree(G, v) for v in G.nodes()) / G.number_of_nodes()

# Reciprocal 2-degree Randić Index
def reciprocal_randic_index_2_degree(G):
    return sum(1 / np.sqrt(two_degree(G, u) * two_degree(G, v)) for u, v in G.edges() if two_degree(G, u) > 0 and two_degree(G, v) > 0)

# Reciprocal 2-degree Sum-connectivity Index
def reciprocal_sum_connectivity_index_2_degree(G):
    return sum(1 / np.sqrt(two_degree(G, u) + two_degree(G, v)) for u, v in G.edges() if (two_degree(G, u) + two_degree(G, v)) > 0)

# 2-degree Hyper-Zagreb Index
def hyper_zagreb_index_2_degree(G):
    return sum(two_degree(G, v) * (two_degree(G, v) - 1) for v in G.nodes() if two_degree(G, v) > 1)

# Reciprocal 2-degree Geometric-Arithmetic Index
def reciprocal_geometric_arithmetic_index_2_degree(G):
    return sum(2 * np.sqrt(two_degree(G, u) * two_degree(G, v)) / (two_degree(G, u) + two_degree(G, v)) for u, v in G.edges() if (two_degree(G, u) + two_degree(G, v)) > 0)


def edge_degree(G, e):
    """Returns the degree of an edge, which is the number of edges sharing an endpoint with the given edge e."""
    u, v = e
    return len(set(G.edges(u)) | set(G.edges(v))) - 1  # Subtract 1 to avoid double-counting the edge itself

def average_edge_degree(G):
    """Computes the average edge degree of the graph G."""
    total_edge_degree = sum(edge_degree(G, e) for e in G.edges())
    return total_edge_degree / G.number_of_edges()

def augmented_average_edge_degree(G):
    """Computes 2 * |E(G)| / (average_edge_degree(G) + 2)."""
    num_edges = G.number_of_edges()
    avg_edge_degree = average_edge_degree(G)
    return 2 * num_edges / (avg_edge_degree + 2)

def inverse_degree_plus_one_sum(G):
    """Computes the sum of the inverse of the degree plus one for all vertices in G."""
    return sum(1 / (G.degree(v) + 1) for v in G.nodes())

def inverse_degree_plus_two_sum(G):
    """Computes the sum of the inverse of the degree plus one for all vertices in G."""
    return sum(1 / (G.degree(v) + 2) for v in G.nodes())

def inverse_edge_degree_plus_one_sum(G):
    """Computes the sum of the inverse of the edge degree plus one for all edges in G."""
    return sum(1 / (edge_degree(G, e) + 1) for e in G.edges())

def inverse_edge_degree_plus_two_sum(G):
    """Computes the sum of the inverse of the edge degree plus one for all edges in G."""
    return sum(1 / (edge_degree(G, e) + 2) for e in G.edges())

# Define the forbidden subgraphs based on Beineke's theorem
def get_forbidden_subgraphs():
    forbidden_graphs = []

    # Claw (K1,3)
    G1 = nx.Graph([(0, 1), (0, 2), (0, 3)])
    forbidden_graphs.append(G1)

    # K5 - e (House)
    G2 = nx.complete_graph(5)
    G2.remove_edge(0, 1)
    forbidden_graphs.append(G2)

    # K3,3 - e (Fork)
    G3 = nx.complete_bipartite_graph(3, 3)
    G3.remove_edge(0, 3)
    forbidden_graphs.append(G3)

    # Diamond (K4 - e)
    G4 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    forbidden_graphs.append(G4)

    # Banner (C4 with an extra vertex connected to two opposite vertices)
    G5 = nx.cycle_graph(4)
    G5.add_node(4)
    G5.add_edges_from([(4, 0), (4, 2)])
    forbidden_graphs.append(G5)

    # C5 (Pentagon)
    G6 = nx.cycle_graph(5)
    forbidden_graphs.append(G6)

    # Butterfly (Two triangles sharing a vertex)
    G7 = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    forbidden_graphs.append(G7)

    # Extended C4 (C4 with an extra vertex connected to 3 consecutive vertices)
    G8 = nx.cycle_graph(4)
    G8.add_node(4)
    G8.add_edges_from([(4, 0), (4, 1), (4, 2)])
    forbidden_graphs.append(G8)

    # Triangle with a pending edge
    G9 = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    forbidden_graphs.append(G9)

    return forbidden_graphs

# Function to check if a graph contains a forbidden subgraph
def contains_forbidden_subgraph(G, forbidden_subgraph):
    GM = nx.algorithms.isomorphism.GraphMatcher(G, forbidden_subgraph)
    return GM.subgraph_is_isomorphic()

# Function to determine if G is a line graph, ignoring C5 as a forbidden subgraph
def is_line_graph_modified(G):
    forbidden_subgraphs = get_forbidden_subgraphs()

    for i, forbidden in enumerate(forbidden_subgraphs, start=1):
        # Ignore Forbidden Subgraph 6 (C5) for line graphs, as it's a valid structure
        if i == 6:
            continue
        if contains_forbidden_subgraph(G, forbidden):
            return False  # Contains a forbidden subgraph, so not a line graph

    return True  # No forbidden subgraphs found, so it's a line graph
