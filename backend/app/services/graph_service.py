# app/services/graph_service.py
from pathlib import Path
from typing import Dict
import asyncio

class GraphService:
    def __init__(self):
        self.cache: Dict[str, dict] = {}
        
    async def load_all_graphs(self):
        """Load and cache all graph files on startup"""
        ppi_dir = Path("ppi")
        
        if not ppi_dir.exists():
            print(f"Warning: PPI directory not found at {ppi_dir}")
            return
            
        graph_files = list(ppi_dir.glob("*_subgraph.txt"))
        print(f"Found {len(graph_files)} graph files to load")
        
        # Process each graph file
        for file_path in graph_files:
            context = file_path.stem.replace("_subgraph", "")
            try:
                graph_data = self._process_graph_file(file_path)
                self.cache[context] = graph_data
                print(f"Loaded graph for context: {context} ({len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges)")
            except Exception as e:
                print(f"Error loading graph for {context}: {e}")
        
        print(f"Successfully loaded {len(self.cache)} graphs into cache")
    
    def _process_graph_file(self, file_path: Path) -> dict:
        """Process a single graph file"""
        degree_map = {}
        edges = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        source, target = parts[0], parts[1]
                        
                        # Count degrees as we read
                        degree_map[source] = degree_map.get(source, 0) + 1
                        degree_map[target] = degree_map.get(target, 0) + 1
                        
                        edges.append({
                            "source": source,
                            "target": target,
                            "weight": 1
                        })
        
        # Create nodes with pre-calculated degrees
        node_list = []
        for node_id, degree in degree_map.items():
            node_list.append({
                "id": node_id,
                "name": node_id,
                "type": "hub" if degree > 5 else "peripheral",
                "degree": degree
            })
        
        # Sort by degree descending
        node_list.sort(key=lambda x: x["degree"], reverse=True)
        
        return {
            "nodes": node_list,
            "edges": edges
        }
    
    def get_graph(self, context: str) -> dict:
        """Get cached graph data for a context"""
        return self.cache.get(context)
    
    def list_contexts(self) -> list:
        """List all available graph contexts"""
        return list(self.cache.keys())

# Global instance
graph_service = GraphService()