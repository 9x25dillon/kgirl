#!/usr/bin/env python3
"""
Emergent Behavior Visualizer
3D fractal and graph visualization for recursive AI patterns
"""

import asyncio
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Try to import plotly for interactive 3D visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available, using matplotlib fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    resolution: int = 100
    color_scheme: str = "viridis"
    node_size_range: Tuple[float, float] = (10, 100)
    edge_width_range: Tuple[float, float] = (0.5, 3.0)
    animation_frames: int = 30
    fractal_iterations: int = 5

@dataclass
class EmergentPattern:
    """Represents an emergent pattern in the visualization"""
    pattern_type: str
    nodes: List[int]
    edges: List[Tuple[int, int]]
    strength: float
    depth: int
    coherence: float
    timestamp: float

class Fractal3DGenerator:
    """Generates 3D fractal patterns for visualization"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
    def generate_3d_mandelbrot(self, resolution: int = 50, max_iter: int = 100) -> np.ndarray:
        """Generate 3D Mandelbrot fractal"""
        x = np.linspace(-2.5, 1.5, resolution)
        y = np.linspace(-2.0, 2.0, resolution)
        z = np.linspace(-1.0, 1.0, resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 3D Mandelbrot calculation
        C = X + 1j * Y + Z * 0.1j  # Add z component
        fractal = np.zeros(C.shape)
        
        for i in range(max_iter):
            mask = np.abs(C) <= 2
            C[mask] = C[mask]**2 + (X[mask] + 1j * Y[mask])
            fractal[mask] = i
        
        return fractal / max_iter
    
    def generate_3d_julia(self, resolution: int = 50, c: complex = -0.7 + 0.27015j) -> np.ndarray:
        """Generate 3D Julia fractal"""
        x = np.linspace(-2.0, 2.0, resolution)
        y = np.linspace(-2.0, 2.0, resolution)
        z = np.linspace(-1.0, 1.0, resolution)
        X, Y, Z = np.meshgrid(x, y, z)
        
        Z_complex = X + 1j * Y + Z * 0.1j
        fractal = np.zeros(Z_complex.shape)
        
        for i in range(max_iter):
            mask = np.abs(Z_complex) <= 2
            Z_complex[mask] = Z_complex[mask]**2 + c
            fractal[mask] = i
        
        return fractal / max_iter
    
    def generate_sierpinski_3d(self, resolution: int = 64, depth: int = 4) -> np.ndarray:
        """Generate 3D Sierpinski tetrahedron"""
        fractal = np.zeros((resolution, resolution, resolution))
        
        def draw_tetrahedron(x, y, z, size, level):
            if level == 0:
                # Draw filled tetrahedron
                for i in range(int(size)):
                    for j in range(int(size - i)):
                        for k in range(int(size - i - j)):
                            if (0 <= x + i < resolution and 
                                0 <= y + j < resolution and 
                                0 <= z + k < resolution):
                                fractal[x + i, y + j, z + k] = 1
            else:
                # Recursively draw smaller tetrahedra
                new_size = size / 2
                draw_tetrahedron(x, y, z, new_size, level - 1)
                draw_tetrahedron(x + new_size, y, z, new_size, level - 1)
                draw_tetrahedron(x + new_size/2, y + new_size, z, new_size, level - 1)
                draw_tetrahedron(x + new_size/2, y + new_size/2, z + new_size, new_size, level - 1)
        
        draw_tetrahedron(0, 0, 0, resolution, depth)
        return fractal

class GraphVisualizer:
    """Visualizes knowledge graphs and emergent patterns"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
    def create_knowledge_graph(self, insights: List[Dict[str, Any]], 
                              embeddings: np.ndarray) -> nx.Graph:
        """Create networkx graph from insights and embeddings"""
        G = nx.Graph()
        
        # Add nodes
        for i, insight in enumerate(insights):
            G.add_node(i, 
                      content=insight.get("text", ""),
                      depth=insight.get("depth", 0),
                      coherence=insight.get("coherence", 0.0),
                      type=insight.get("type", "unknown"))
        
        # Add edges based on similarity
        if len(embeddings) > 1:
            similarity_matrix = self._calculate_similarity_matrix(embeddings)
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = similarity_matrix[i, j]
                    if similarity > 0.7:  # Threshold for edge creation
                        G.add_edge(i, j, weight=similarity)
        
        return G
    
    def _calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix"""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms!=0)
        
        # Calculate cosine similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix
    
    def detect_emergent_patterns(self, graph: nx.Graph) -> List[EmergentPattern]:
        """Detect emergent patterns in the graph"""
        patterns = []
        
        # Detect communities/clusters
        communities = self._detect_communities(graph)
        for i, community in enumerate(communities):
            if len(community) >= 3:  # Minimum community size
                pattern = EmergentPattern(
                    pattern_type="community",
                    nodes=list(community),
                    edges=[(u, v) for u, v in graph.edges() if u in community and v in community],
                    strength=len(community) / len(graph.nodes()),
                    depth=1,
                    coherence=self._calculate_community_coherence(graph, community),
                    timestamp=time.time()
                )
                patterns.append(pattern)
        
        # Detect cycles
        cycles = list(nx.simple_cycles(graph.to_directed()))
        for i, cycle in enumerate(cycles):
            if len(cycle) >= 3:
                pattern = EmergentPattern(
                    pattern_type="cycle",
                    nodes=cycle,
                    edges=[(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))],
                    strength=len(cycle) / len(graph.nodes()),
                    depth=2,
                    coherence=0.8,  # Cycles often indicate strong patterns
                    timestamp=time.time()
                )
                patterns.append(pattern)
        
        # Detect star patterns (high-degree nodes)
        degree_centrality = nx.degree_centrality(graph)
        high_degree_nodes = [node for node, centrality in degree_centrality.items() 
                           if centrality > 0.3]
        
        for node in high_degree_nodes:
            neighbors = list(graph.neighbors(node))
            if len(neighbors) >= 3:
                pattern = EmergentPattern(
                    pattern_type="star",
                    nodes=[node] + neighbors,
                    edges=[(node, neighbor) for neighbor in neighbors],
                    strength=len(neighbors) / len(graph.nodes()),
                    depth=1,
                    coherence=degree_centrality[node],
                    timestamp=time.time()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_communities(self, graph: nx.Graph) -> List[List[int]]:
        """Detect communities using simple clustering"""
        try:
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(graph))
            return [list(community) for community in communities]
        except ImportError:
            # Fallback: simple connected components
            return [list(component) for component in nx.connected_components(graph)]
    
    def _calculate_community_coherence(self, graph: nx.Graph, community: List[int]) -> float:
        """Calculate coherence of a community"""
        if len(community) < 2:
            return 0.0
        
        # Calculate average edge weight within community
        total_weight = 0
        edge_count = 0
        
        for u in community:
            for v in community:
                if u < v and graph.has_edge(u, v):
                    total_weight += graph[u][v].get('weight', 1.0)
                    edge_count += 1
        
        if edge_count == 0:
            return 0.0
        
        return total_weight / edge_count

class EmergentVisualizer:
    """Main visualizer for emergent behavior patterns"""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.fractal_generator = Fractal3DGenerator(self.config)
        self.graph_visualizer = GraphVisualizer(self.config)
        self.patterns_history: List[EmergentPattern] = []
        
    def visualize_3d_fractal(self, fractal_type: str = "mandelbrot", 
                           save_path: str = None) -> None:
        """Visualize 3D fractal patterns"""
        if PLOTLY_AVAILABLE:
            self._visualize_3d_fractal_plotly(fractal_type, save_path)
        else:
            self._visualize_3d_fractal_matplotlib(fractal_type, save_path)
    
    def _visualize_3d_fractal_plotly(self, fractal_type: str, save_path: str = None):
        """Interactive 3D fractal visualization with Plotly"""
        if fractal_type == "mandelbrot":
            fractal = self.fractal_generator.generate_3d_mandelbrot(30)
        elif fractal_type == "julia":
            fractal = self.fractal_generator.generate_3d_julia(30)
        elif fractal_type == "sierpinski":
            fractal = self.fractal_generator.generate_sierpinski_3d(32, 3)
        else:
            fractal = self.fractal_generator.generate_3d_mandelbrot(30)
        
        # Create 3D scatter plot
        x, y, z = np.where(fractal > 0.1)
        values = fractal[x, y, z]
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=values,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f'Value: {v:.3f}' for v in values],
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'3D {fractal_type.title()} Fractal Pattern',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"3D fractal visualization saved to {save_path}")
        
        fig.show()
    
    def _visualize_3d_fractal_matplotlib(self, fractal_type: str, save_path: str = None):
        """3D fractal visualization with matplotlib"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if fractal_type == "mandelbrot":
            fractal = self.fractal_generator.generate_3d_mandelbrot(20)
        elif fractal_type == "julia":
            fractal = self.fractal_generator.generate_3d_julia(20)
        elif fractal_type == "sierpinski":
            fractal = self.fractal_generator.generate_sierpinski_3d(16, 3)
        else:
            fractal = self.fractal_generator.generate_3d_mandelbrot(20)
        
        # Create 3D scatter plot
        x, y, z = np.where(fractal > 0.1)
        values = fractal[x, y, z]
        
        scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=20, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D {fractal_type.title()} Fractal Pattern')
        
        plt.colorbar(scatter)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D fractal visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_knowledge_graph(self, insights: List[Dict[str, Any]], 
                                 embeddings: np.ndarray, 
                                 save_path: str = None) -> None:
        """Visualize knowledge graph with emergent patterns"""
        # Create graph
        graph = self.graph_visualizer.create_knowledge_graph(insights, embeddings)
        
        # Detect patterns
        patterns = self.graph_visualizer.detect_emergent_patterns(graph)
        self.patterns_history.extend(patterns)
        
        if PLOTLY_AVAILABLE:
            self._visualize_graph_plotly(graph, patterns, save_path)
        else:
            self._visualize_graph_matplotlib(graph, patterns, save_path)
    
    def _visualize_graph_plotly(self, graph: nx.Graph, patterns: List[EmergentPattern], 
                               save_path: str = None):
        """Interactive graph visualization with Plotly"""
        # Get node positions using spring layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Prepare node data
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [f"Node {node}<br>Content: {graph.nodes[node].get('content', '')[:50]}...<br>Depth: {graph.nodes[node].get('depth', 0)}" 
                    for node in graph.nodes()]
        node_sizes = [graph.nodes[node].get('coherence', 0.5) * 20 + 10 for node in graph.nodes()]
        node_colors = [graph.nodes[node].get('depth', 0) for node in graph.nodes()]
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f"Node {node}" for node in graph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_sizes,
                color=node_colors,
                colorbar=dict(
                    thickness=15,
                    x=1.05,
                    len=0.5,
                    title="Depth"
                ),
                line=dict(width=2, color='black')
            )
        )
        
        # Create pattern traces
        pattern_traces = []
        for i, pattern in enumerate(patterns):
            pattern_x = [pos[node][0] for node in pattern.nodes if node in graph.nodes()]
            pattern_y = [pos[node][1] for node in pattern.nodes if node in graph.nodes()]
            
            pattern_trace = go.Scatter(
                x=pattern_x, y=pattern_y,
                mode='markers',
                marker=dict(
                    size=15,
                    color=f'rgba(255, 0, 0, {pattern.strength})',
                    symbol='star'
                ),
                name=f"{pattern.pattern_type} (strength: {pattern.strength:.2f})",
                showlegend=True
            )
            pattern_traces.append(pattern_trace)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace] + pattern_traces)
        
        fig.update_layout(
            title='Knowledge Graph with Emergent Patterns',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size = Coherence, Color = Depth",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="black", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Graph visualization saved to {save_path}")
        
        fig.show()
    
    def _visualize_graph_matplotlib(self, graph: nx.Graph, patterns: List[EmergentPattern], 
                                  save_path: str = None):
        """Graph visualization with matplotlib"""
        plt.figure(figsize=(12, 8))
        
        # Get node positions
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
        
        # Draw nodes with different colors for depth
        node_colors = [graph.nodes[node].get('depth', 0) for node in graph.nodes()]
        node_sizes = [graph.nodes[node].get('coherence', 0.5) * 200 + 50 for node in graph.nodes()]
        
        nodes = nx.draw_networkx_nodes(graph, pos, 
                                     node_color=node_colors,
                                     node_size=node_sizes,
                                     cmap='viridis',
                                     alpha=0.8)
        
        # Draw pattern highlights
        for pattern in patterns:
            pattern_nodes = [node for node in pattern.nodes if node in graph.nodes()]
            if pattern_nodes:
                nx.draw_networkx_nodes(graph, pos, 
                                     nodelist=pattern_nodes,
                                     node_color='red',
                                     node_size=300,
                                     alpha=0.6)
        
        # Add labels
        nx.draw_networkx_labels(graph, pos, font_size=8)
        
        plt.title('Knowledge Graph with Emergent Patterns')
        plt.colorbar(nodes, label='Depth')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_emergence_timeline(self, save_path: str = None) -> None:
        """Visualize emergence patterns over time"""
        if not self.patterns_history:
            logger.warning("No patterns to visualize")
            return
        
        # Group patterns by timestamp
        timestamps = [p.timestamp for p in self.patterns_history]
        pattern_types = [p.pattern_type for p in self.patterns_history]
        strengths = [p.strength for p in self.patterns_history]
        
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Create traces for each pattern type
            for pattern_type in set(pattern_types):
                type_timestamps = [t for t, pt in zip(timestamps, pattern_types) if pt == pattern_type]
                type_strengths = [s for s, pt in zip(strengths, pattern_types) if pt == pattern_type]
                
                fig.add_trace(go.Scatter(
                    x=type_timestamps,
                    y=type_strengths,
                    mode='markers+lines',
                    name=pattern_type,
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title='Emergence Patterns Over Time',
                xaxis_title='Time',
                yaxis_title='Pattern Strength',
                hovermode='closest'
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Timeline visualization saved to {save_path}")
            
            fig.show()
        else:
            # Matplotlib fallback
            plt.figure(figsize=(12, 6))
            
            for pattern_type in set(pattern_types):
                type_timestamps = [t for t, pt in zip(timestamps, pattern_types) if pt == pattern_type]
                type_strengths = [s for s, pt in zip(strengths, pattern_types) if pt == pattern_type]
                
                plt.plot(type_timestamps, type_strengths, 'o-', label=pattern_type, markersize=6)
            
            plt.xlabel('Time')
            plt.ylabel('Pattern Strength')
            plt.title('Emergence Patterns Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Timeline visualization saved to {save_path}")
            
            plt.show()
    
    def get_emergence_statistics(self) -> Dict[str, Any]:
        """Get statistics about emergent patterns"""
        if not self.patterns_history:
            return {"message": "No patterns detected"}
        
        pattern_types = [p.pattern_type for p in self.patterns_history]
        strengths = [p.strength for p in self.patterns_history]
        coherences = [p.coherence for p in self.patterns_history]
        
        type_counts = {}
        for pattern_type in pattern_types:
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        return {
            "total_patterns": len(self.patterns_history),
            "pattern_type_distribution": type_counts,
            "average_strength": float(np.mean(strengths)),
            "max_strength": float(np.max(strengths)),
            "average_coherence": float(np.mean(coherences)),
            "max_coherence": float(np.max(coherences)),
            "emergence_rate": len(self.patterns_history) / max(1, (max([p.timestamp for p in self.patterns_history]) - min([p.timestamp for p in self.patterns_history])))
        }

# Demo function
async def demo_emergent_visualizer():
    """Demonstrate emergent behavior visualization"""
    config = VisualizationConfig(
        resolution=50,
        fractal_iterations=3
    )
    
    visualizer = EmergentVisualizer(config)
    
    # Generate sample insights and embeddings
    insights = [
        {"text": "Quantum computing uses superposition", "depth": 0, "coherence": 0.8, "type": "quantum"},
        {"text": "Neural networks learn patterns", "depth": 1, "coherence": 0.7, "type": "neural"},
        {"text": "Recursive systems create emergence", "depth": 2, "coherence": 0.9, "type": "recursive"},
        {"text": "Fractal patterns repeat infinitely", "depth": 1, "coherence": 0.6, "type": "fractal"},
        {"text": "Consciousness emerges from complexity", "depth": 3, "coherence": 0.8, "type": "consciousness"},
    ]
    
    embeddings = np.random.randn(len(insights), 128)
    
    # Visualize knowledge graph
    print("Visualizing knowledge graph...")
    visualizer.visualize_knowledge_graph(insights, embeddings, "knowledge_graph.html")
    
    # Visualize 3D fractals
    print("Visualizing 3D fractals...")
    visualizer.visualize_3d_fractal("mandelbrot", "mandelbrot_3d.html")
    visualizer.visualize_3d_fractal("sierpinski", "sierpinski_3d.html")
    
    # Get emergence statistics
    stats = visualizer.get_emergence_statistics()
    print("Emergence Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(demo_emergent_visualizer())