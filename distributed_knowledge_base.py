#!/usr/bin/env python3
"""
Distributed Knowledge Base
FAISS/SQLite-based central knowledge hub for collective intelligence network
"""

import asyncio
import json
import logging
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import threading
from pathlib import Path

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, using numpy fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Represents a knowledge node in the distributed system"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    coherence_score: float = 0.0

@dataclass
class KnowledgeBaseConfig:
    """Configuration for the distributed knowledge base"""
    db_path: str = "knowledge_base.db"
    faiss_index_path: str = "faiss_index"
    embedding_dimension: int = 768
    max_nodes: int = 100000
    similarity_threshold: float = 0.7
    sync_interval: float = 30.0  # seconds
    backup_interval: float = 3600.0  # 1 hour

class SQLiteKnowledgeStore:
    """SQLite-based persistent storage for knowledge nodes"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            cursor = self.connection.cursor()
            
            # Knowledge nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    source TEXT,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 0,
                    coherence_score REAL DEFAULT 0.0
                )
            """)
            
            # Node relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS node_relationships (
                    source_id TEXT,
                    target_id TEXT,
                    relationship_type TEXT,
                    strength REAL,
                    timestamp REAL,
                    PRIMARY KEY (source_id, target_id, relationship_type),
                    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(id),
                    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(id)
                )
            """)
            
            # System metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    timestamp REAL
                )
            """)
            
            self.connection.commit()
            logger.info(f"SQLite knowledge store initialized: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite store: {e}")
            return False
    
    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a knowledge node to the store"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Serialize embedding
                embedding_blob = node.embedding.tobytes()
                metadata_json = json.dumps(node.metadata)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge_nodes 
                    (id, content, embedding, metadata, source, timestamp, access_count, coherence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.id, node.content, embedding_blob, metadata_json,
                    node.source, node.timestamp, node.access_count, node.coherence_score
                ))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a knowledge node by ID"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("SELECT * FROM knowledge_nodes WHERE id = ?", (node_id,))
                row = cursor.fetchone()
                
                if row:
                    # Deserialize embedding
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    metadata = json.loads(row['metadata'])
                    
                    return KnowledgeNode(
                        id=row['id'],
                        content=row['content'],
                        embedding=embedding,
                        metadata=metadata,
                        source=row['source'],
                        timestamp=row['timestamp'],
                        access_count=row['access_count'],
                        coherence_score=row['coherence_score']
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def search_nodes(self, query: str, limit: int = 10) -> List[KnowledgeNode]:
        """Search nodes by content"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT * FROM knowledge_nodes 
                    WHERE content LIKE ? 
                    ORDER BY coherence_score DESC, timestamp DESC 
                    LIMIT ?
                """, (f"%{query}%", limit))
                
                nodes = []
                for row in cursor.fetchall():
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                    metadata = json.loads(row['metadata'])
                    
                    node = KnowledgeNode(
                        id=row['id'],
                        content=row['content'],
                        embedding=embedding,
                        metadata=metadata,
                        source=row['source'],
                        timestamp=row['timestamp'],
                        access_count=row['access_count'],
                        coherence_score=row['coherence_score']
                    )
                    nodes.append(node)
                
                return nodes
                
        except Exception as e:
            logger.error(f"Failed to search nodes: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Node count
                cursor.execute("SELECT COUNT(*) FROM knowledge_nodes")
                node_count = cursor.fetchone()[0]
                
                # Average coherence
                cursor.execute("SELECT AVG(coherence_score) FROM knowledge_nodes")
                avg_coherence = cursor.fetchone()[0] or 0.0
                
                # Source distribution
                cursor.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM knowledge_nodes 
                    GROUP BY source 
                    ORDER BY count DESC
                """)
                source_dist = dict(cursor.fetchall())
                
                return {
                    "total_nodes": node_count,
                    "average_coherence": float(avg_coherence),
                    "source_distribution": source_dist
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

class FAISSKnowledgeIndex:
    """FAISS-based vector index for similarity search"""
    
    def __init__(self, index_path: str, dimension: int = 768):
        self.index_path = index_path
        self.dimension = dimension
        self.index = None
        self.node_ids = []
        self.lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using numpy fallback")
            return False
        
        try:
            # Try to load existing index
            if Path(self.index_path).exists():
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            return False
    
    def add_embeddings(self, embeddings: np.ndarray, node_ids: List[str]) -> bool:
        """Add embeddings to the index"""
        if not self.index:
            return False
        
        try:
            with self.lock:
                # Ensure embeddings are float32
                if embeddings.dtype != np.float32:
                    embeddings = embeddings.astype(np.float32)
                
                # Add to index
                self.index.add(embeddings)
                self.node_ids.extend(node_ids)
                
                logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """Search for similar vectors"""
        if not self.index or self.index.ntotal == 0:
            return np.array([]), []
        
        try:
            with self.lock:
                # Ensure query vector is float32 and 2D
                if query_vector.dtype != np.float32:
                    query_vector = query_vector.astype(np.float32)
                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)
                
                # Search
                distances, indices = self.index.search(query_vector, k)
                
                # Map indices to node IDs
                result_ids = [self.node_ids[i] for i in indices[0] if i < len(self.node_ids)]
                
                return distances[0], result_ids
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return np.array([]), []
    
    def save_index(self) -> bool:
        """Save index to disk"""
        if not self.index:
            return False
        
        try:
            with self.lock:
                faiss.write_index(self.index, self.index_path)
                logger.info(f"FAISS index saved to {self.index_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

class DistributedKnowledgeBase:
    """Main distributed knowledge base system"""
    
    def __init__(self, config: KnowledgeBaseConfig = None):
        self.config = config or KnowledgeBaseConfig()
        self.sqlite_store = SQLiteKnowledgeStore(self.config.db_path)
        self.faiss_index = FAISSKnowledgeIndex(self.config.faiss_index_path, self.config.embedding_dimension)
        self.sync_timer = None
        self.backup_timer = None
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize the knowledge base system"""
        logger.info("Initializing distributed knowledge base...")
        
        # Initialize SQLite store
        if not self.sqlite_store.initialize():
            return False
        
        # Initialize FAISS index
        if not self.faiss_index.initialize():
            logger.warning("FAISS index not available, continuing with SQLite only")
        
        # Start background tasks
        self.is_running = True
        self.sync_timer = asyncio.create_task(self._sync_loop())
        self.backup_timer = asyncio.create_task(self._backup_loop())
        
        logger.info("Distributed knowledge base initialized successfully")
        return True
    
    async def add_knowledge_node(self, content: str, embedding: np.ndarray, 
                                source: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new knowledge node"""
        node_id = f"node_{int(time.time() * 1000)}_{hash(content) % 10000}"
        
        if metadata is None:
            metadata = {}
        
        # Calculate coherence score (simplified)
        coherence_score = min(1.0, np.linalg.norm(embedding) / 10.0)
        
        node = KnowledgeNode(
            id=node_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
            source=source,
            coherence_score=coherence_score
        )
        
        # Add to SQLite store
        if self.sqlite_store.add_node(node):
            # Add to FAISS index
            if self.faiss_index.index:
                self.faiss_index.add_embeddings(
                    embedding.reshape(1, -1), 
                    [node_id]
                )
            
            logger.info(f"Added knowledge node: {node_id}")
            return node_id
        else:
            raise Exception(f"Failed to add knowledge node: {node_id}")
    
    async def search_knowledge(self, query: str, query_embedding: np.ndarray = None, 
                              k: int = 5) -> List[KnowledgeNode]:
        """Search knowledge base"""
        results = []
        
        # Text-based search
        text_results = self.sqlite_store.search_nodes(query, limit=k)
        results.extend(text_results)
        
        # Vector-based search (if FAISS available and query embedding provided)
        if query_embedding is not None and self.faiss_index.index:
            distances, node_ids = self.faiss_index.search(query_embedding, k)
            
            for node_id in node_ids:
                node = self.sqlite_store.get_node(node_id)
                if node:
                    results.append(node)
        
        # Remove duplicates and sort by coherence
        seen_ids = set()
        unique_results = []
        for node in results:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                unique_results.append(node)
        
        # Sort by coherence score
        unique_results.sort(key=lambda x: x.coherence_score, reverse=True)
        
        return unique_results[:k]
    
    async def get_related_nodes(self, node_id: str, k: int = 5) -> List[KnowledgeNode]:
        """Get nodes related to a specific node"""
        node = self.sqlite_store.get_node(node_id)
        if not node:
            return []
        
        # Search using the node's embedding
        distances, related_ids = self.faiss_index.search(node.embedding.reshape(1, -1), k + 1)
        
        # Remove the node itself from results
        related_ids = [nid for nid in related_ids if nid != node_id]
        
        # Get the actual nodes
        related_nodes = []
        for nid in related_ids[:k]:
            related_node = self.sqlite_store.get_node(nid)
            if related_node:
                related_nodes.append(related_node)
        
        return related_nodes
    
    async def sync_with_network(self, network_nodes: List[KnowledgeNode]) -> Dict[str, Any]:
        """Sync with other nodes in the network"""
        sync_stats = {
            "nodes_received": len(network_nodes),
            "nodes_added": 0,
            "nodes_updated": 0,
            "sync_time": time.time()
        }
        
        for node in network_nodes:
            existing = self.sqlite_store.get_node(node.id)
            if existing:
                # Update if newer or higher coherence
                if (node.timestamp > existing.timestamp or 
                    node.coherence_score > existing.coherence_score):
                    self.sqlite_store.add_node(node)
                    sync_stats["nodes_updated"] += 1
            else:
                # Add new node
                self.sqlite_store.add_node(node)
                if self.faiss_index.index:
                    self.faiss_index.add_embeddings(
                        node.embedding.reshape(1, -1), 
                        [node.id]
                    )
                sync_stats["nodes_added"] += 1
        
        logger.info(f"Network sync completed: {sync_stats}")
        return sync_stats
    
    async def _sync_loop(self):
        """Background sync loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.sync_interval)
                # In a real implementation, this would sync with other network nodes
                logger.debug("Sync loop tick")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
    
    async def _backup_loop(self):
        """Background backup loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.backup_interval)
                await self.backup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
    
    async def backup(self) -> bool:
        """Create backup of the knowledge base"""
        try:
            timestamp = int(time.time())
            backup_path = f"{self.config.db_path}.backup_{timestamp}"
            
            # Copy SQLite database
            import shutil
            shutil.copy2(self.config.db_path, backup_path)
            
            # Save FAISS index
            if self.faiss_index.index:
                faiss_backup_path = f"{self.config.faiss_index_path}.backup_{timestamp}"
                self.faiss_index.save_index()
                shutil.copy2(self.config.faiss_index_path, faiss_backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        sqlite_stats = self.sqlite_store.get_statistics()
        
        stats = {
            "sqlite": sqlite_stats,
            "faiss_available": FAISS_AVAILABLE,
            "faiss_index_size": self.faiss_index.index.ntotal if self.faiss_index.index else 0,
            "config": {
                "embedding_dimension": self.config.embedding_dimension,
                "max_nodes": self.config.max_nodes,
                "similarity_threshold": self.config.similarity_threshold
            }
        }
        
        return stats
    
    async def close(self):
        """Close the knowledge base system"""
        self.is_running = False
        
        # Cancel background tasks
        if self.sync_timer:
            self.sync_timer.cancel()
        if self.backup_timer:
            self.backup_timer.cancel()
        
        # Save FAISS index
        if self.faiss_index.index:
            self.faiss_index.save_index()
        
        # Close SQLite connection
        if self.sqlite_store.connection:
            self.sqlite_store.connection.close()
        
        logger.info("Distributed knowledge base closed")

# Demo function
async def demo_distributed_knowledge_base():
    """Demonstrate distributed knowledge base capabilities"""
    config = KnowledgeBaseConfig(
        db_path="demo_knowledge.db",
        faiss_index_path="demo_faiss_index",
        embedding_dimension=128
    )
    
    kb = DistributedKnowledgeBase(config)
    
    # Initialize
    if not await kb.initialize():
        logger.error("Failed to initialize knowledge base")
        return
    
    # Add some sample knowledge nodes
    sample_nodes = [
        ("Quantum computing uses superposition", np.random.randn(128), "quantum_physics"),
        ("Neural networks learn from data", np.random.randn(128), "machine_learning"),
        ("Recursive systems create emergence", np.random.randn(128), "cognitive_science"),
    ]
    
    for content, embedding, source in sample_nodes:
        node_id = await kb.add_knowledge_node(content, embedding, source)
        print(f"Added node: {node_id}")
    
    # Search knowledge
    search_results = await kb.search_knowledge("quantum", k=3)
    print(f"\nSearch results for 'quantum': {len(search_results)} nodes found")
    
    for node in search_results:
        print(f"  - {node.content[:50]}... (coherence: {node.coherence_score:.3f})")
    
    # Get statistics
    stats = kb.get_statistics()
    print(f"\nKnowledge Base Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Close
    await kb.close()

if __name__ == "__main__":
    asyncio.run(demo_distributed_knowledge_base())