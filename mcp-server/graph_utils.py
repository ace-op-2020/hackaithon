"""
Graph Utilities for Trellix Knowledge Graph
==========================================

Utility functions for graph analysis, visualization, and query operations.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GraphMetrics:
    """Metrics for knowledge graph analysis"""
    node_count: int
    edge_count: int
    connected_components: int
    average_degree: float
    density: float
    diameter: Optional[int]
    clustering_coefficient: float
    centrality_measures: Dict[str, Dict[str, float]]

class GraphAnalyzer:
    """Analyze knowledge graph structure and properties"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def compute_metrics(self) -> GraphMetrics:
        """Compute comprehensive graph metrics"""
        try:
            # Basic metrics
            node_count = self.graph.number_of_nodes()
            edge_count = self.graph.number_of_edges()
            
            # Convert to undirected for some metrics
            undirected_graph = self.graph.to_undirected()
            
            # Connectivity metrics
            connected_components = nx.number_connected_components(undirected_graph)
            
            # Degree metrics
            degrees = [d for n, d in undirected_graph.degree()]
            average_degree = sum(degrees) / len(degrees) if degrees else 0
            
            # Density
            density = nx.density(undirected_graph)
            
            # Diameter (for largest connected component)
            diameter = None
            if connected_components > 0:
                largest_cc = max(nx.connected_components(undirected_graph), key=len)
                subgraph = undirected_graph.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    diameter = nx.diameter(subgraph)
            
            # Clustering coefficient
            clustering_coefficient = nx.average_clustering(undirected_graph)
            
            # Centrality measures
            centrality_measures = self._compute_centrality_measures()
            
            return GraphMetrics(
                node_count=node_count,
                edge_count=edge_count,
                connected_components=connected_components,
                average_degree=average_degree,
                density=density,
                diameter=diameter,
                clustering_coefficient=clustering_coefficient,
                centrality_measures=centrality_measures
            )
            
        except Exception as e:
            logger.error(f"Error computing graph metrics: {e}")
            return GraphMetrics(0, 0, 0, 0.0, 0.0, None, 0.0, {})
    
    def _compute_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Compute various centrality measures"""
        centrality_measures = {}
        
        try:
            # Convert to undirected for centrality calculations
            undirected_graph = self.graph.to_undirected()
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(undirected_graph)
            centrality_measures['degree'] = dict(sorted(
                degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(undirected_graph, k=100)
            centrality_measures['betweenness'] = dict(sorted(
                betweenness_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
            
            # Closeness centrality
            closeness_centrality = nx.closeness_centrality(undirected_graph)
            centrality_measures['closeness'] = dict(sorted(
                closeness_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
            
            # PageRank
            pagerank = nx.pagerank(self.graph)
            centrality_measures['pagerank'] = dict(sorted(
                pagerank.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
            
        except Exception as e:
            logger.error(f"Error computing centrality measures: {e}")
        
        return centrality_measures
    
    def find_communities(self) -> List[List[str]]:
        """Find communities in the graph using the Louvain algorithm"""
        try:
            import networkx.algorithms.community as nx_comm
            
            undirected_graph = self.graph.to_undirected()
            communities = nx_comm.louvain_communities(undirected_graph)
            return [list(community) for community in communities]
            
        except Exception as e:
            logger.error(f"Error finding communities: {e}")
            return []
    
    def get_node_neighborhoods(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get the neighborhood of a specific node"""
        try:
            if node_id not in self.graph:
                return {}
            
            # Get subgraph within specified depth
            subgraph_nodes = set([node_id])
            for d in range(depth):
                new_nodes = set()
                for node in subgraph_nodes:
                    new_nodes.update(self.graph.neighbors(node))
                    new_nodes.update(self.graph.predecessors(node))
                subgraph_nodes.update(new_nodes)
            
            subgraph = self.graph.subgraph(subgraph_nodes)
            
            return {
                'center_node': node_id,
                'nodes': list(subgraph.nodes()),
                'edges': list(subgraph.edges()),
                'node_count': subgraph.number_of_nodes(),
                'edge_count': subgraph.number_of_edges()
            }
            
        except Exception as e:
            logger.error(f"Error getting node neighborhood: {e}")
            return {}

class GraphVisualizer:
    """Visualize knowledge graphs using various methods"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def create_interactive_visualization(self, output_path: str = "knowledge_graph.html",
                                       height: str = "800px", width: str = "100%"):
        """Create an interactive HTML visualization using pyvis"""
        try:
            from pyvis.network import Network
            
            # Create pyvis network
            net = Network(height=height, width=width, directed=True)
            
            # Add nodes
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get('type', 'Entity')
                node_name = node_data.get('name', str(node_id))
                
                # Color code by node type
                color_map = {
                    'Product': '#ff6b6b',
                    'Feature': '#4ecdc4',
                    'Component': '#45b7d1',
                    'Process': '#96ceb4',
                    'Person': '#ffeaa7',
                    'Organization': '#dda0dd'
                }
                color = color_map.get(node_type, '#gray')
                
                net.add_node(
                    str(node_id),
                    label=node_name,
                    title=f"Type: {node_type}\nID: {node_id}",
                    color=color
                )
            
            # Add edges
            for source, target, edge_data in self.graph.edges(data=True):
                relationship_type = edge_data.get('type', 'RELATED_TO')
                confidence = edge_data.get('confidence', 0.5)
                
                net.add_edge(
                    str(source),
                    str(target),
                    label=relationship_type,
                    title=f"Relationship: {relationship_type}\nConfidence: {confidence:.2f}",
                    width=max(1, confidence * 5)
                )
            
            # Configure physics
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Save visualization
            net.save_graph(output_path)
            logger.info(f"Interactive visualization saved to {output_path}")
            
        except ImportError:
            logger.error("pyvis not available. Install with: pip install pyvis")
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def export_to_formats(self, base_filename: str = "knowledge_graph"):
        """Export graph to various formats"""
        try:
            # Export to GraphML (for Gephi, Cytoscape)
            nx.write_graphml(self.graph, f"{base_filename}.graphml")
            
            # Export to GML
            nx.write_gml(self.graph, f"{base_filename}.gml")
            
            # Export to JSON
            graph_data = nx.node_link_data(self.graph)
            with open(f"{base_filename}.json", 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            # Export adjacency matrix to CSV
            import pandas as pd
            adj_matrix = nx.adjacency_matrix(self.graph)
            df = pd.DataFrame(adj_matrix.todense(), 
                            index=list(self.graph.nodes()), 
                            columns=list(self.graph.nodes()))
            df.to_csv(f"{base_filename}_adjacency.csv")
            
            logger.info(f"Graph exported to multiple formats with base name: {base_filename}")
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")

class GraphQueryEngine:
    """Query engine for knowledge graph operations"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            if source in self.graph and target in self.graph:
                return nx.shortest_path(self.graph, source, target)
            return None
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return None
    
    def find_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Find all nodes of a specific type"""
        nodes = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == node_type:
                nodes.append({
                    'id': node_id,
                    'data': node_data
                })
        return nodes
    
    def find_related_nodes(self, node_id: str, relationship_type: Optional[str] = None,
                          max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find nodes related to a given node"""
        if node_id not in self.graph:
            return []
        
        related_nodes = []
        visited = set([node_id])
        
        def explore_neighbors(current_node, depth):
            if depth > max_depth:
                return
            
            # Outgoing edges
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    if relationship_type is None or any(
                        ed.get('type') == relationship_type for ed in edge_data.values()
                    ):
                        related_nodes.append({
                            'id': neighbor,
                            'data': self.graph.nodes[neighbor],
                            'path_length': depth + 1,
                            'relationship': edge_data
                        })
                        visited.add(neighbor)
                        explore_neighbors(neighbor, depth + 1)
            
            # Incoming edges
            for neighbor in self.graph.predecessors(current_node):
                if neighbor not in visited:
                    edge_data = self.graph.get_edge_data(neighbor, current_node)
                    if relationship_type is None or any(
                        ed.get('type') == relationship_type for ed in edge_data.values()
                    ):
                        related_nodes.append({
                            'id': neighbor,
                            'data': self.graph.nodes[neighbor],
                            'path_length': depth + 1,
                            'relationship': edge_data
                        })
                        visited.add(neighbor)
                        explore_neighbors(neighbor, depth + 1)
        
        explore_neighbors(node_id, 0)
        return related_nodes
    
    def search_nodes_by_name(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search nodes by name using fuzzy matching"""
        import difflib
        
        results = []
        query_lower = query.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            node_name = node_data.get('name', str(node_id)).lower()
            
            # Exact match
            if query_lower in node_name:
                similarity = 1.0 if query_lower == node_name else 0.8
                results.append({
                    'id': node_id,
                    'data': node_data,
                    'similarity': similarity
                })
            else:
                # Fuzzy match
                similarity = difflib.SequenceMatcher(None, query_lower, node_name).ratio()
                if similarity > 0.6:  # Threshold for fuzzy matching
                    results.append({
                        'id': node_id,
                        'data': node_data,
                        'similarity': similarity
                    })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """Get statistics about nodes and relationships"""
        node_types = {}
        relationship_types = {}
        
        # Count node types
        for _, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'Unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count relationship types
        for _, _, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get('type', 'Unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_relationships': self.graph.number_of_edges(),
            'node_types': node_types,
            'relationship_types': relationship_types,
            'most_connected_nodes': self._get_most_connected_nodes(10)
        }
    
    def _get_most_connected_nodes(self, limit: int) -> List[Dict[str, Any]]:
        """Get the most connected nodes"""
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for node_id, centrality in sorted_nodes[:limit]:
            node_data = self.graph.nodes[node_id]
            result.append({
                'id': node_id,
                'name': node_data.get('name', str(node_id)),
                'type': node_data.get('type', 'Unknown'),
                'degree_centrality': centrality,
                'degree': self.graph.degree(node_id)
            })
        
        return result

# Utility functions
def merge_graphs(graph1: nx.MultiDiGraph, graph2: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Merge two knowledge graphs"""
    merged_graph = nx.MultiDiGraph()
    
    # Add nodes from both graphs
    for graph in [graph1, graph2]:
        for node_id, node_data in graph.nodes(data=True):
            if node_id in merged_graph:
                # Merge node data
                existing_data = merged_graph.nodes[node_id]
                merged_data = {**existing_data, **node_data}
                merged_graph.nodes[node_id].update(merged_data)
            else:
                merged_graph.add_node(node_id, **node_data)
    
    # Add edges from both graphs
    for graph in [graph1, graph2]:
        for source, target, edge_data in graph.edges(data=True):
            merged_graph.add_edge(source, target, **edge_data)
    
    return merged_graph

def validate_graph(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Validate graph structure and report issues"""
    issues = []
    statistics = {}
    
    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        issues.append(f"Found {len(isolated_nodes)} isolated nodes")
    
    # Check for self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        issues.append(f"Found {len(self_loops)} self-loops")
    
    # Check for nodes without required attributes
    nodes_missing_type = []
    nodes_missing_name = []
    
    for node_id, node_data in graph.nodes(data=True):
        if 'type' not in node_data:
            nodes_missing_type.append(node_id)
        if 'name' not in node_data:
            nodes_missing_name.append(node_id)
    
    if nodes_missing_type:
        issues.append(f"Found {len(nodes_missing_type)} nodes missing 'type' attribute")
    if nodes_missing_name:
        issues.append(f"Found {len(nodes_missing_name)} nodes missing 'name' attribute")
    
    # Graph statistics
    statistics = {
        'node_count': graph.number_of_nodes(),
        'edge_count': graph.number_of_edges(),
        'isolated_nodes': len(isolated_nodes),
        'self_loops': len(self_loops),
        'nodes_missing_type': len(nodes_missing_type),
        'nodes_missing_name': len(nodes_missing_name)
    }
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'statistics': statistics
    }