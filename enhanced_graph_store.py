#!/usr/bin/env python3
"""
Enhanced Graph Store with Numbskull Integration
===============================================

Knowledge graph system integrated with Numbskull embeddings:
- Node and edge management with embedded representations
- Semantic relationship discovery
- Graph-based retrieval and reasoning
- Embedding-enhanced graph traversal

Author: Assistant
License: MIT
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

try:
    from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
    NUMBSKULL_AVAILABLE = True
except ImportError:
    NUMBSKULL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Node in the knowledge graph"""
    id: str
    label: str
    content: str
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class GraphEdge:
    """Edge in the knowledge graph"""
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class EnhancedGraphStore:
    """
    Knowledge graph with Numbskull embedding integration
    
    Provides semantic graph operations with embedding-based reasoning
    """
    
    def __init__(
        self,
        use_numbskull: bool = True,
        numbskull_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize enhanced graph store
        
        Args:
            use_numbskull: Use Numbskull for node embeddings
            numbskull_config: Configuration for Numbskull pipeline
        """
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, Set[str]] = {}  # node_id -> set of connected node_ids
        
        # Initialize Numbskull pipeline
        if use_numbskull and NUMBSKULL_AVAILABLE:
            config = HybridConfig(**(numbskull_config or {}))
            self.numbskull = HybridEmbeddingPipeline(config)
            logger.info("✅ Enhanced graph store with Numbskull embeddings")
        else:
            self.numbskull = None
            logger.warning("⚠️  Graph store without Numbskull")
    
    async def add_node(
        self,
        id: str,
        label: str,
        content: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add node to graph
        
        Args:
            id: Unique node identifier
            label: Node label/type
            content: Node content for embedding
            properties: Optional node properties
        
        Returns:
            Success status
        """
        try:
            # Generate embedding for node
            embedding = None
            if self.numbskull:
                result = await self.numbskull.embed(content)
                embedding = result["fused_embedding"]
            
            # Create node
            node = GraphNode(
                id=id,
                label=label,
                content=content,
                embedding=embedding,
                properties=properties or {}
            )
            
            self.nodes[id] = node
            if id not in self.adjacency:
                self.adjacency[id] = set()
            
            logger.debug(f"Added node {id} ({label})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add node {id}: {e}")
            return False
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add edge to graph
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relationship type
            weight: Edge weight
            properties: Optional edge properties
        
        Returns:
            Success status
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes {source_id} or {target_id} not found")
            return False
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            properties=properties or {}
        )
        
        self.edges.append(edge)
        self.adjacency[source_id].add(target_id)
        
        logger.debug(f"Added edge {source_id} --[{relation}]--> {target_id}")
        return True
    
    async def find_similar_nodes(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5
    ) -> List[Tuple[GraphNode, float]]:
        """
        Find nodes similar to query
        
        Args:
            query: Query text
            top_k: Number of results
            threshold: Similarity threshold
        
        Returns:
            List of (node, similarity) tuples
        """
        if not self.numbskull:
            logger.warning("Cannot find similar nodes without Numbskull")
            return []
        
        # Generate query embedding
        result = await self.numbskull.embed(query)
        query_embedding = result["fused_embedding"]
        
        # Compute similarities
        similarities = []
        for node in self.nodes.values():
            if node.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, node.embedding)
                if similarity >= threshold:
                    similarities.append((node, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_neighbors(self, node_id: str, depth: int = 1) -> Set[str]:
        """
        Get neighbors of a node up to specified depth
        
        Args:
            node_id: Starting node ID
            depth: Traversal depth
        
        Returns:
            Set of neighbor node IDs
        """
        if node_id not in self.nodes:
            return set()
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(depth):
            next_level = set()
            for nid in current_level:
                if nid in self.adjacency:
                    next_level.update(self.adjacency[nid])
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors
    
    def get_subgraph(self, node_ids: List[str]) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Extract subgraph containing specified nodes
        
        Args:
            node_ids: List of node IDs
        
        Returns:
            (nodes, edges) in subgraph
        """
        node_set = set(node_ids)
        nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        edges = [
            edge for edge in self.edges
            if edge.source_id in node_set and edge.target_id in node_set
        ]
        return nodes, edges
    
    def get_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3
    ) -> List[List[str]]:
        """
        Find paths between two nodes
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path length
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        paths = []
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current == target_id:
                paths.append(path.copy())
                return
            
            if current in self.adjacency:
                for neighbor in self.adjacency[current]:
                    if neighbor not in path:  # Avoid cycles
                        path.append(neighbor)
                        dfs(neighbor, path, depth + 1)
                        path.pop()
        
        dfs(source_id, [source_id], 0)
        return paths
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "avg_degree": sum(len(neighbors) for neighbors in self.adjacency.values()) / max(len(self.nodes), 1),
            "numbskull_enabled": self.numbskull is not None,
            "nodes_with_embeddings": sum(1 for node in self.nodes.values() if node.embedding is not None)
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull:
            await self.numbskull.close()


async def demo_enhanced_graph_store():
    """Demonstration of enhanced graph store"""
    print("\n" + "=" * 70)
    print("ENHANCED GRAPH STORE DEMO")
    print("=" * 70)
    
    # Create graph
    graph = EnhancedGraphStore(
        use_numbskull=NUMBSKULL_AVAILABLE,
        numbskull_config={
            "use_semantic": False,
            "use_mathematical": False,
            "use_fractal": True,
            "cache_embeddings": True
        }
    )
    
    # Add nodes
    print("\nBuilding knowledge graph...")
    nodes = [
        ("ai", "Technology", "Artificial intelligence and machine learning"),
        ("ml", "Technology", "Machine learning algorithms and models"),
        ("nn", "Technology", "Neural networks and deep learning"),
        ("python", "Language", "Python programming language"),
        ("data", "Concept", "Data analysis and processing"),
    ]
    
    for id, label, content in nodes:
        await graph.add_node(id, label, content)
    
    # Add edges
    graph.add_edge("ai", "ml", "includes")
    graph.add_edge("ml", "nn", "uses")
    graph.add_edge("python", "ml", "implements")
    graph.add_edge("data", "ml", "feeds")
    graph.add_edge("nn", "python", "coded_in")
    
    print(f"✅ Created graph with {len(nodes)} nodes and {len(graph.edges)} edges")
    
    # Find similar nodes
    query = "deep learning and neural computation"
    print(f"\nFinding nodes similar to: '{query}'")
    similar = await graph.find_similar_nodes(query, top_k=3)
    
    for i, (node, score) in enumerate(similar, 1):
        print(f"  {i}. [{score:.3f}] {node.id} ({node.label}): {node.content}")
    
    # Find paths
    print(f"\nFinding paths from 'ai' to 'python':")
    paths = graph.get_paths("ai", "python", max_depth=3)
    for i, path in enumerate(paths, 1):
        path_str = " -> ".join(path)
        print(f"  {i}. {path_str}")
    
    # Get neighbors
    print(f"\nNeighbors of 'ml' (depth=1):")
    neighbors = graph.get_neighbors("ml", depth=1)
    for nid in neighbors:
        node = graph.nodes[nid]
        print(f"  - {nid} ({node.label})")
    
    # Stats
    print("\nGraph Statistics:")
    stats = graph.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    await graph.close()
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_enhanced_graph_store())

