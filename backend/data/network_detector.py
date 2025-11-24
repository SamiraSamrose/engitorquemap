## **File: backend/data/network_detector.py** (Community Detection)

"""
Network Detector - Community Detection
Detects driver communities/connections using network analysis
Advanced feature for analyzing driver relationships and competition patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("networkx not installed. Community detection limited.")

logger = logging.getLogger(__name__)


class CommunityDetector:
    """
    Detects communities and connections in driver network
    """
    
    def __init__(self):
        self.graph = None
        self.communities = {}
    
    async def detect_communities(self, track_name: str) -> Dict:
        """
        Detect driver communities based on racing patterns
        
        Returns network analysis including:
        - Driver clusters
        - Competition patterns
        - Performance correlations
        """
        if not NETWORKX_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "NetworkX not installed"
            }
        
        logger.info(f"Detecting communities for {track_name}")
        
        # Build driver interaction graph
        self.graph = await self._build_driver_graph(track_name)
        
        # Detect communities using Louvain algorithm
        communities = self._detect_communities_louvain()
        
        # Analyze competition patterns
        competition = self._analyze_competition_patterns()
        
        # Calculate network metrics
        metrics = self._calculate_network_metrics()
        
        return {
            "track": track_name,
            "communities": communities,
            "competition_patterns": competition,
            "network_metrics": metrics,
            "graph_data": self._export_graph_data()
        }
    
    async def _build_driver_graph(self, track_name: str) -> nx.Graph:
        """Build network graph of driver interactions"""
        from pathlib import Path
        from config import settings
        
        G = nx.Graph()
        
        # Load timing data to get driver interactions
        data_dir = Path(settings.PROCESSED_DATA_DIR)
        timing_files = list(data_dir.glob(f"{track_name}*timing*.parquet"))
        
        if not timing_files:
            # Create dummy graph for demonstration
            return self._create_dummy_graph()
        
        # Load timing data
        timing_df = pd.read_parquet(timing_files[0])
        
        # Add drivers as nodes
        if 'driver_id' in timing_df.columns:
            drivers = timing_df['driver_id'].unique()
            for driver in drivers:
                G.add_node(driver, driver_id=driver)
        
        # Add edges based on race battles (close lap times)
        if 'lap_time' in timing_df.columns and 'driver_id' in timing_df.columns:
            # Group by lap
            for lap_num in timing_df['lap'].unique() if 'lap' in timing_df.columns else [1]:
                lap_data = timing_df[timing_df['lap'] == lap_num] if 'lap' in timing_df.columns else timing_df
                
                # Find drivers with close lap times (within 0.5s)
                for i, driver1 in enumerate(lap_data['driver_id'].unique()):
                    time1 = lap_data[lap_data['driver_id'] == driver1]['lap_time'].mean()
                    
                    for driver2 in lap_data['driver_id'].unique()[i+1:]:
                        time2 = lap_data[lap_data['driver_id'] == driver2]['lap_time'].mean()
                        
                        if abs(time1 - time2) < 0.5:  # Close battle
                            # Add edge with weight based on closeness
                            weight = 1.0 / (abs(time1 - time2) + 0.01)
                            
                            if G.has_edge(driver1, driver2):
                                G[driver1][driver2]['weight'] += weight
                            else:
                                G.add_edge(driver1, driver2, weight=weight)
        
        return G
    
    def _create_dummy_graph(self) -> nx.Graph:
        """Create dummy graph for demonstration"""
        G = nx.Graph()
        
        # Add 20 drivers
        drivers = [f"D{i:03d}" for i in range(1, 21)]
        G.add_nodes_from(drivers)
        
        # Add random edges (battles)
        for i in range(len(drivers)):
            for j in range(i+1, len(drivers)):
                if np.random.random() > 0.7:  # 30% chance of connection
                    weight = np.random.uniform(0.5, 3.0)
                    G.add_edge(drivers[i], drivers[j], weight=weight)
        
        return G
    
    def _detect_communities_louvain(self) -> List[Dict]:
        """Detect communities using Louvain algorithm"""
        if not self.graph:
            return []
        
        try:
            # Use Louvain community detection
            communities_generator = community.greedy_modularity_communities(
                self.graph, 
                weight='weight'
            )
            
            community_list = list(communities_generator)
            
            # Format communities
            formatted_communities = []
            for idx, comm in enumerate(community_list):
                formatted_communities.append({
                    "community_id": idx,
                    "size": len(comm),
                    "members": list(comm),
                    "description": f"Competition group {idx + 1}"
                })
            
            logger.info(f"Detected {len(formatted_communities)} communities")
            return formatted_communities
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return []
    
    def _analyze_competition_patterns(self) -> Dict:
        """Analyze competition patterns in the network"""
        if not self.graph:
            return {}
        
        patterns = {
            "close_battles": [],
            "dominant_pairs": [],
            "isolated_drivers": []
        }
        
        # Find close battles (high weight edges)
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 0)
            if weight > 2.0:
                patterns["close_battles"].append({
                    "driver1": u,
                    "driver2": v,
                    "intensity": float(weight)
                })
        
        # Find isolated drivers (low connectivity)
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree < 2:
                patterns["isolated_drivers"].append(str(node))
        
        return patterns
    
    def _calculate_network_metrics(self) -> Dict:
        """Calculate network-level metrics"""
        if not self.graph:
            return {}
        
        metrics = {}
        
        try:
            # Density
            metrics['density'] = nx.density(self.graph)
            
            # Average clustering coefficient
            metrics['clustering_coefficient'] = nx.average_clustering(self.graph)
            
            # Number of components
            metrics['num_components'] = nx.number_connected_components(self.graph)
            
            # Degree centrality
            centrality = nx.degree_centrality(self.graph)
            metrics['most_central_driver'] = max(centrality, key=centrality.get)
            metrics['avg_centrality'] = float(np.mean(list(centrality.values())))
            
        except Exception as e:
            logger.error(f"Failed to calculate network metrics: {e}")
        
        return metrics
    
    def _export_graph_data(self) -> Dict:
        """Export graph data for visualization"""
        if not self.graph:
            return {}
        
        # Convert to format suitable for frontend visualization
        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                "id": str(node),
                "label": str(node),
                "degree": self.graph.degree(node)
            })
        
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "weight": float(data.get('weight', 1.0))
            })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
