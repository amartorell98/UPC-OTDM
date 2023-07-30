#!/usr/bin/env python3
"""
Modified graph implementation to compute clusters based on the minimum spanning tree heuristic.
"""
from functools import total_ordering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import warnings

class Vertex:
    """ 
    Class representing a vertex in a graph.
    
    Attributes:
        id (int): The identifier for the vertex.
    """
    def __init__(self, id=None):
        self.id = id

    def __call__(self):
        return self.id

@total_ordering
class Edge:
    """
    Class representing an edge in a graph.
    Edges can be compared by means of their weight.
    
    Attributes:
        u (Vertex): The first vertex connected by the edge.
        v (Vertex): The second vertex connected by the edge.
        w (int): The weight of the edge.

    """
    def __init__(self, u, v, weight=None):
        self.u = u
        self.v = v
        self.w = weight

    def __call__(self):
        return self.u, self.v, self.w

    def __lt__(self, other):
        return self.w < other.w

    def __eq__(self, other):
        return self.w == other.w

class Graph:
    """
    Class representing a graph data structure.
    """
    def __init__(self, vertices=[], edges=[]):
        self.V = vertices
        self.E = edges
    
    def add_edge(self, edge):
        """
        Add an edge to the graph.
        
        Args:
            edge (Edge): The edge to be added.
        """
        self.E.append(edge)

    # Auxiliary
    def build_from_distance_matrix(self, D):
        """
        Build a graph from a distance matrix.
        
        Args:
            D (list): The distance matrix to build the graph from.
            
        Returns:
            list: The list of edges in the graph.
        """
        self.D = D
        self.hashEdges = {}
        self.V = [Vertex(id=i) for i in range(D.shape[0])]
        for i in range(D.shape[0]): # Iterate through data points (nodes)
            for j in range(i, D.shape[1]): # Iterate through distances
                d = D[i][j]
                edge = Edge(self.V[i], self.V[j], weight=d)
                self.E.append(edge)

    def get_adjacency_matrix(self, E, directed=False, weights=False):
        """
        Get the adjacency matrix of the graph.
        
        Args:
            E (list): The list of edges in the graph.
            directed (bool, optional): If true, the edges are directed.
            weights (bool, optional): If True, the adjacency matrix will include weights.
        """
        A = [[0]*len(self.V) for _ in range(len(self.V))]
        for edge in E:
            u, v, w = edge()
            if not weights:
                w = 1
            A[u.id][v.id] = w
            if not directed:
                A[v.id][u.id] = w
        return A

    def get_connected_components(self, A, directed=False):
        """
        Get the connected components of the graph.
        
        Args:
            A (list): The adjacency matrix of the graph.
            directed (bool, optional): If True, the graph will be treated as directed.
            
        Returns:
            list: The connected components of the graph.
        """
        return connected_components(csr_matrix(A), directed=directed)

    def cut_k_edges_from_mst(self, k, mst):
        """
        Cut k edges from the minimum spanning tree of the graph.
        
        Args:
            k (int): The number of edges to cut.
            mst (list): The minimum spanning tree of the graph.
            
        Returns:
            list: The minimum spanning tree with k edges removed.
        """
        return sorted(mst, reverse=False)[:-k]
        
    # Kruskal's Algorithm
    def kruskal_alg(self):
        """
        Find the minimum spanning tree of the graph using Kruskal's algorithm.
        
        Returns:
            list: The minimum spanning tree of the graph.
        """
        try:
            mst = []
            i, e = 0, 0
            G = sorted(self.E, reverse=False)
            parent = []
            rank = []
            for vertex in range(len(self.V)):
                parent.append(vertex)
                rank.append(0)
            while e < (len(self.V) - 1):
                u, v, w = G[i]()
                i += 1
                x = self._find_cycle(parent, u.id)
                y = self._find_cycle(parent, v.id)
                if x != y:
                    e += 1
                    mst.append(G[i-1])
                    self._union(parent, rank, x, y)                
            self.kruskal_mst = mst
            return mst

        except Exception as e:
            warnings.warn(f"The graph might not be connected. Excepction: {e}")

    def _find_cycle(self, parent, i):
        """
        Find the parent of vertex i in the graph.
        
        Args:
            parent (list): The parent array for the graph.
            i (int): The vertex to find the parent of.
            
        Returns:
            int: The parent of vertex i.
        """
        if parent[i] == i:
            return i
        return self._find_cycle(parent, parent[i])

    def _union(self, parent, rank, x, y):
        """
        Combine two sets of vertices into one.
        
        Args:
            parent (list): The parent array for the graph.
            rank (list): The rank array for the graph.
            x (int): The first vertex to combine.
            y (int): The second vertex to combine.
        """
        xroot = self._find_cycle(parent, x)
        yroot = self._find_cycle(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # Prim's Algorithm
    def prim_alg(self):
        """
        Find the minimum spanning tree of the graph using Prim's algorithm.
        
        Args:
            start (Vertex): The vertex where the algorithm will start.
            
        Returns:
            list: The minimum spanning tree of the graph.
        """
        G = self.get_adjacency_matrix(self.E, directed=False, weights=True)

        V = len(self.V)
        selected = [0]*V
        no_edge = 0
        selected[0] = True
        mst = []
        while (no_edge < V - 1):
            minimum = 1e5
            x, y = 0, 0
            for i in range(V):
                if selected[i]:
                    for j in range(V):
                        if ((not selected[j]) and G[i][j]):  
                            if minimum > G[i][j]:
                                minimum = G[i][j]
                                x = i
                                y = j
            mst.append(Edge(self.V[x], self.V[y], G[x][y]))
            selected[y] = True
            no_edge += 1

        self.prim_mst = mst
        return mst