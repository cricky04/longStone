'''
for artificial data(theorical research)
'''

import torch
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph

# Barabasi-Albert graph generator
def BAGenerator(num_nodes, num_edges):
    '''
    input
        num_nodes: number of nodes
        num_edges: number of edges
    output
        data: torch_geometry.Data
    '''
    # TODO: add node features
    baGraph = BAGraph(num_nodes, num_edges)    
    
    return baGraph()

def ERGenerator(num_nodes, prob_edges):
    '''
    input
        num_nodes: number of nodes
        prob_edges: probability of edges
    output
        data: torch_geometry.Data
    '''
    # TODO: add node features
    erGraph = ERGraph(num_nodes, prob_edges)
    
    return erGraph() 
    