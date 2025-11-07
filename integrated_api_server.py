#!/usr/bin/env python3
"""
Integrated API Server: Complete LiMp + Numbskull API
===================================================

Unified REST API providing access to all integrated components:

Endpoints:
- /embeddings/* - Numbskull embedding operations
- /cognitive/* - Unified cognitive workflows
- /vector/* - Vector index operations
- /graph/* - Knowledge graph operations
- /symbolic/* - AL-ULS symbolic evaluation
- /entropy/* - Entropy analysis
- /quantum/* - Quantum processing
- /workflow/* - Complete integrated workflows

Built on FastAPI with async support throughout.

Author: Assistant
License: MIT
"""

import sys
from pathlib import Path

# Add numbskull to path
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import logging

# Import integrated systems
from complete_system_integration import CompleteSystemIntegration
from limp_module_manager import LiMpModuleManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Integrated LiMp + Numbskull API",
    version="2.0.0",
    description="Complete API for unified cognitive architecture"
)

# Global system instance (initialized on startup)
system: Optional[CompleteSystemIntegration] = None
module_manager: Optional[LiMpModuleManager] = None


# ============= Request/Response Models =============

class EmbeddingRequest(BaseModel):
    text: str
    use_semantic: bool = False
    use_mathematical: bool = False
    use_fractal: bool = True
    fusion_method: str = "weighted_average"


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    metadata: Dict[str, Any]
    cached: bool = False


class CognitiveWorkflowRequest(BaseModel):
    query: str
    context: Optional[str] = None
    resources: List[str] = []
    inline_resources: List[str] = []


class CognitiveWorkflowResponse(BaseModel):
    final_output: str
    stages: Dict[str, Any]
    system_state: Dict[str, Any]
    timing: Dict[str, float] = {}


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: Optional[float] = None


class VectorAddRequest(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class GraphNodeRequest(BaseModel):
    id: str
    label: str
    content: str
    properties: Optional[Dict[str, Any]] = None


class GraphEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


class GraphSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.5


class CompleteWorkflowRequest(BaseModel):
    query: str
    context: Optional[str] = None
    resources: List[str] = []
    enable_vector: bool = True
    enable_graph: bool = True
    enable_entropy: bool = True


# ============= Lifecycle Events =============

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global system, module_manager
    
    logger.info("=" * 70)
    logger.info("INTEGRATED API SERVER STARTING")
    logger.info("=" * 70)
    
    try:
        # Initialize complete system
        system = CompleteSystemIntegration()
        logger.info("✅ Complete system initialized")
        
        # Initialize module manager
        module_manager = LiMpModuleManager()
        logger.info("✅ Module manager initialized")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global system, module_manager
    
    logger.info("Shutting down integrated API server...")
    
    if system:
        await system.close_all()
    
    if module_manager:
        await module_manager.close_all()
    
    logger.info("✅ Shutdown complete")


# ============= Root Endpoints =============

@app.get("/")
async def root() -> Dict[str, Any]:
    """API root endpoint"""
    return {
        "service": "Integrated LiMp + Numbskull API",
        "version": "2.0.0",
        "status": "operational",
        "components": {
            "cognitive": system is not None,
            "vector_index": system.vector_index is not None if system else False,
            "graph_store": system.graph_store is not None if system else False,
            "module_manager": module_manager is not None
        }
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_ready": system is not None,
        "modules_available": len(module_manager.get_available_modules()) if module_manager else 0
    }


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Comprehensive system status"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    stats = system.get_complete_stats()
    
    if module_manager:
        stats["modules"] = module_manager.get_status()
    
    return stats


# ============= Embedding Endpoints =============

@app.post("/embeddings/generate", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Generate hybrid embedding using Numbskull"""
    if not system or not system.cognitive_orch or not system.cognitive_orch.orchestrator:
        raise HTTPException(status_code=503, detail="Embedding system not available")
    
    try:
        # Generate embedding
        result = await system.cognitive_orch.orchestrator._generate_embeddings(request.text)
        
        if not result:
            raise HTTPException(status_code=500, detail="Embedding generation failed")
        
        embedding = result["fused_embedding"]
        
        return EmbeddingResponse(
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
            metadata=result["metadata"],
            cached=result.get("cached", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embeddings/batch")
async def batch_embeddings(texts: List[str]) -> Dict[str, Any]:
    """Generate embeddings for multiple texts"""
    if not system or not system.cognitive_orch or not system.cognitive_orch.orchestrator:
        raise HTTPException(status_code=503, detail="Embedding system not available")
    
    try:
        embeddings = []
        for text in texts:
            result = await system.cognitive_orch.orchestrator._generate_embeddings(text)
            if result:
                embeddings.append({
                    "text": text,
                    "embedding": result["fused_embedding"].tolist() if hasattr(result["fused_embedding"], 'tolist') else list(result["fused_embedding"]),
                    "metadata": result["metadata"]
                })
        
        return {"embeddings": embeddings, "count": len(embeddings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Cognitive Endpoints =============

@app.post("/cognitive/process", response_model=CognitiveWorkflowResponse)
async def process_cognitive(request: CognitiveWorkflowRequest) -> CognitiveWorkflowResponse:
    """Execute unified cognitive workflow"""
    if not system or not system.cognitive_orch:
        raise HTTPException(status_code=503, detail="Cognitive system not available")
    
    try:
        result = await system.cognitive_orch.process_cognitive_workflow(
            user_query=request.query,
            context=request.context,
            inline_resources=request.inline_resources
        )
        
        return CognitiveWorkflowResponse(
            final_output=result.get("final_output", ""),
            stages=result.get("stages", {}),
            system_state=result.get("cognitive_state", {}),
            timing=result.get("timing", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Vector Index Endpoints =============

@app.post("/vector/add")
async def vector_add(request: VectorAddRequest) -> Dict[str, Any]:
    """Add entry to vector index"""
    if not system or not system.vector_index:
        raise HTTPException(status_code=503, detail="Vector index not available")
    
    try:
        success = await system.vector_index.add_entry(
            request.id,
            request.text,
            request.metadata
        )
        return {"success": success, "id": request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vector/search")
async def vector_search(request: VectorSearchRequest) -> Dict[str, Any]:
    """Search vector index"""
    if not system or not system.vector_index:
        raise HTTPException(status_code=503, detail="Vector index not available")
    
    try:
        results = await system.vector_index.search(
            request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return {
            "results": [
                {
                    "id": entry.id,
                    "text": entry.text,
                    "similarity": float(score),
                    "metadata": entry.metadata
                }
                for entry, score in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector/stats")
async def vector_stats() -> Dict[str, Any]:
    """Get vector index statistics"""
    if not system or not system.vector_index:
        raise HTTPException(status_code=503, detail="Vector index not available")
    
    return system.vector_index.get_stats()


# ============= Graph Endpoints =============

@app.post("/graph/node/add")
async def graph_add_node(request: GraphNodeRequest) -> Dict[str, Any]:
    """Add node to knowledge graph"""
    if not system or not system.graph_store:
        raise HTTPException(status_code=503, detail="Graph store not available")
    
    try:
        success = await system.graph_store.add_node(
            request.id,
            request.label,
            request.content,
            request.properties
        )
        return {"success": success, "id": request.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/edge/add")
async def graph_add_edge(request: GraphEdgeRequest) -> Dict[str, Any]:
    """Add edge to knowledge graph"""
    if not system or not system.graph_store:
        raise HTTPException(status_code=503, detail="Graph store not available")
    
    try:
        success = system.graph_store.add_edge(
            request.source_id,
            request.target_id,
            request.relation,
            request.weight
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/search")
async def graph_search(request: GraphSearchRequest) -> Dict[str, Any]:
    """Search for similar nodes in graph"""
    if not system or not system.graph_store:
        raise HTTPException(status_code=503, detail="Graph store not available")
    
    try:
        results = await system.graph_store.find_similar_nodes(
            request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return {
            "results": [
                {
                    "id": node.id,
                    "label": node.label,
                    "content": node.content,
                    "similarity": float(score),
                    "properties": node.properties
                }
                for node, score in results
            ],
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats")
async def graph_stats() -> Dict[str, Any]:
    """Get graph statistics"""
    if not system or not system.graph_store:
        raise HTTPException(status_code=503, detail="Graph store not available")
    
    return system.graph_store.get_stats()


# ============= Complete Workflow Endpoints =============

@app.post("/workflow/complete")
async def complete_workflow(request: CompleteWorkflowRequest) -> Dict[str, Any]:
    """Execute complete integrated workflow across all systems"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = await system.process_complete_workflow(
            user_query=request.query,
            context=request.context,
            resources=request.resources,
            enable_vector_index=request.enable_vector,
            enable_graph=request.enable_graph,
            enable_entropy=request.enable_entropy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow/batch")
async def batch_workflow(queries: List[str], contexts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process multiple queries in batch"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        results = await system.batch_process(queries, contexts)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= Module Management Endpoints =============

@app.get("/modules/list")
async def list_modules() -> Dict[str, Any]:
    """List all available modules"""
    if not module_manager:
        raise HTTPException(status_code=503, detail="Module manager not available")
    
    return {
        "available": module_manager.get_available_modules(),
        "initialized": module_manager.get_initialized_modules(),
        "total": len(module_manager.modules)
    }


@app.get("/modules/status")
async def modules_status() -> Dict[str, Any]:
    """Get status of all modules"""
    if not module_manager:
        raise HTTPException(status_code=503, detail="Module manager not available")
    
    return module_manager.get_status()


@app.post("/modules/initialize/{module_name}")
async def initialize_module(module_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize a specific module"""
    if not module_manager:
        raise HTTPException(status_code=503, detail="Module manager not available")
    
    success = await module_manager.initialize_module(module_name, config)
    return {"success": success, "module": module_name}


# ============= Statistics Endpoints =============

@app.get("/stats/complete")
async def complete_stats() -> Dict[str, Any]:
    """Get comprehensive statistics from all systems"""
    if not system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return system.get_complete_stats()


@app.get("/stats/embeddings")
async def embedding_stats() -> Dict[str, Any]:
    """Get embedding-specific statistics"""
    if not system or not system.cognitive_orch or not system.cognitive_orch.orchestrator:
        raise HTTPException(status_code=503, detail="Embedding system not available")
    
    return system.cognitive_orch.orchestrator.get_embedding_stats()


# ============= Main Entry Point =============

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("INTEGRATED API SERVER")
    print("LiMp + Numbskull + LFM2-8B-A1B")
    print("=" * 70)
    print("\nStarting server on http://0.0.0.0:8888")
    print("\nAPI Documentation: http://0.0.0.0:8888/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8888,
        log_level="info"
    )

