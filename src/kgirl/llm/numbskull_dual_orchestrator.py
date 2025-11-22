#!/usr/bin/env python3
"""
Numbskull-Enhanced Dual LLM Orchestration System
=================================================

Integrates the numbskull embedding pipeline with dual LLM orchestration:
- Numbskull: Hybrid embeddings (semantic, mathematical, fractal)
- Local LLM (LFM2-8B-A1B): Final inference and decision making
- Remote LLM: Resource-only summarization and structuring

This orchestrator generates rich embeddings for resources before
passing them to the dual LLM system for enhanced contextual understanding.

Author: Assistant
License: MIT
"""

import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import base dual LLM orchestrator
from dual_llm_orchestrator import (
    DualLLMOrchestrator,
    LocalLLM,
    ResourceLLM,
    HTTPConfig,
    OrchestratorSettings,
    BaseLLM,
    HAS_REQUESTS
)

# Add numbskull to path if needed
numbskull_path = Path("/home/kill/numbskull")
if numbskull_path.exists() and str(numbskull_path) not in sys.path:
    sys.path.insert(0, str(numbskull_path))

# Import numbskull pipeline
try:
    from advanced_embedding_pipeline import (
        HybridEmbeddingPipeline,
        HybridConfig,
        SemanticConfig,
        MathematicalConfig,
        FractalConfig
    )
    NUMBSKULL_AVAILABLE = True
except ImportError as e:
    NUMBSKULL_AVAILABLE = False
    HybridEmbeddingPipeline = None
    logging.warning(f"Numbskull pipeline not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NumbskullOrchestratorSettings(OrchestratorSettings):
    """Extended settings with numbskull configuration"""
    # Numbskull pipeline settings
    use_numbskull: bool = True
    use_semantic: bool = True
    use_mathematical: bool = True
    use_fractal: bool = True
    fusion_method: str = "weighted_average"  # "weighted_average", "concatenation", "attention"
    
    # Embedding weights
    semantic_weight: float = 0.4
    mathematical_weight: float = 0.3
    fractal_weight: float = 0.3
    
    # Embedding processing
    embed_resources: bool = True
    embed_user_prompt: bool = False
    max_embedding_cache_size: int = 1000
    
    # Integration mode
    embedding_enhancement: str = "metadata"  # "metadata", "similarity", "full_vectors"


class NumbskullDualOrchestrator(DualLLMOrchestrator):
    """
    Enhanced orchestrator that integrates numbskull embeddings
    with dual LLM workflow for superior contextual understanding.
    """
    
    def __init__(
        self, 
        local: LocalLLM, 
        resource: ResourceLLM, 
        settings: NumbskullOrchestratorSettings,
        numbskull_config: Optional[HybridConfig] = None
    ):
        super().__init__(local, resource, settings)
        self.settings: NumbskullOrchestratorSettings = settings
        
        # Initialize numbskull pipeline
        self.numbskull_pipeline = None
        self.embedding_cache = {}
        self.embedding_stats = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "embedding_time": 0.0,
            "components_used": {}
        }
        
        if settings.use_numbskull and NUMBSKULL_AVAILABLE:
            try:
                self._initialize_numbskull(numbskull_config)
            except Exception as e:
                logger.error(f"Failed to initialize numbskull pipeline: {e}")
                logger.info("Continuing without numbskull embeddings")
    
    def _initialize_numbskull(self, config: Optional[HybridConfig] = None):
        """Initialize the numbskull embedding pipeline"""
        if config is None:
            # Create default configuration from settings
            config = HybridConfig(
                use_semantic=self.settings.use_semantic,
                use_mathematical=self.settings.use_mathematical,
                use_fractal=self.settings.use_fractal,
                fusion_method=self.settings.fusion_method,
                semantic_weight=self.settings.semantic_weight,
                mathematical_weight=self.settings.mathematical_weight,
                fractal_weight=self.settings.fractal_weight,
                parallel_processing=True,
                cache_embeddings=True,
                timeout=60.0
            )
        
        self.numbskull_pipeline = HybridEmbeddingPipeline(config)
        logger.info("✅ Numbskull pipeline initialized with hybrid embedding support")
    
    async def _generate_embeddings(self, text: str) -> Optional[Dict[str, Any]]:
        """Generate hybrid embeddings for text using numbskull pipeline"""
        if not self.numbskull_pipeline:
            return None
        
        try:
            # Check cache
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self.embedding_cache:
                self.embedding_stats["cache_hits"] += 1
                return self.embedding_cache[cache_key]
            
            # Generate embeddings
            start_time = time.time()
            embedding_result = await self.numbskull_pipeline.embed(text)
            embedding_time = time.time() - start_time
            
            # Update stats
            self.embedding_stats["total_embeddings"] += 1
            self.embedding_stats["embedding_time"] += embedding_time
            
            for component in embedding_result["metadata"]["components_used"]:
                self.embedding_stats["components_used"][component] = \
                    self.embedding_stats["components_used"].get(component, 0) + 1
            
            # Cache result (limit cache size)
            if len(self.embedding_cache) < self.settings.max_embedding_cache_size:
                self.embedding_cache[cache_key] = embedding_result
            
            return embedding_result
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
    
    def _format_embedding_metadata(self, embedding_result: Dict[str, Any]) -> str:
        """Format embedding metadata for inclusion in prompts"""
        if not embedding_result:
            return ""
        
        metadata = embedding_result.get("metadata", {})
        components = metadata.get("components_used", [])
        dim = metadata.get("embedding_dim", 0)
        processing_time = metadata.get("processing_time", 0.0)
        
        meta_text = f"""
EMBEDDING ANALYSIS:
- Components: {', '.join(components)}
- Dimension: {dim}
- Processing Time: {processing_time:.3f}s
- Cached: {embedding_result.get('cached', False)}
"""
        
        if self.settings.embedding_enhancement == "full_vectors":
            # Include actual embedding vectors (truncated)
            embeddings = embedding_result.get("embeddings", {})
            for component, vector in embeddings.items():
                if vector is not None:
                    vector_str = str(vector[:5].tolist() if hasattr(vector, 'tolist') else vector[:5])
                    meta_text += f"- {component.capitalize()}: {vector_str}...\n"
        
        return meta_text.strip()
    
    async def compose_with_embeddings(
        self, 
        user_prompt: str, 
        resource_paths: List[str], 
        inline_resources: List[str]
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Enhanced compose that generates embeddings before summarization
        
        Returns:
            Tuple of (final_prompt, resource_summary, embedding_results)
        """
        # Load resources
        resource_text = self._load_resources(resource_paths, inline_resources)
        
        # Generate embeddings if enabled
        embedding_result = None
        if self.settings.embed_resources and self.numbskull_pipeline:
            logger.info("Generating numbskull embeddings for resources...")
            embedding_result = await self._generate_embeddings(resource_text)
        
        # Format embedding metadata
        embedding_metadata = ""
        if embedding_result:
            embedding_metadata = self._format_embedding_metadata(embedding_result)
            logger.info(f"Embeddings generated: {embedding_result['metadata']['components_used']}")
        
        # Create enhanced resource prompt for summarization
        if embedding_metadata:
            resource_prompt = f"""INPUT RESOURCES:
{resource_text}

{embedding_metadata}

TASK: Summarize/structure ONLY the content above, taking into account the embedding analysis."""
        else:
            resource_prompt = f"INPUT RESOURCES:\n{resource_text}\n\nTASK: Summarize/structure ONLY the content above."
        
        # Resource LLM summarization
        resource_summary = self.resource.generate(
            resource_prompt,
            temperature=0.2,
            max_tokens=self.settings.max_tokens
        )
        
        # Create final prompt for local LLM (LFM2-8B-A1B)
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{resource_summary}\n\n"
        )
        
        if embedding_metadata and self.settings.embedding_enhancement != "none":
            final_prompt += f"=== EMBEDDING CONTEXT ===\n{embedding_metadata}\n\n"
        
        final_prompt += (
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.settings.style}. Be clear and directly actionable."
        )
        
        return final_prompt, resource_summary, embedding_result
    
    async def run_with_embeddings(
        self, 
        user_prompt: str, 
        resource_paths: List[str], 
        inline_resources: List[str]
    ) -> Dict[str, Any]:
        """
        Execute full dual LLM orchestration with numbskull embeddings
        
        Returns enhanced result dictionary with embedding information
        """
        try:
            # Compose with embeddings
            final_prompt, summary, embedding_result = await self.compose_with_embeddings(
                user_prompt, resource_paths, inline_resources
            )
            
            # Local LLM (LFM2-8B-A1B) generates final answer
            logger.info("Sending to LFM2-8B-A1B for final inference...")
            answer = self.local.generate(
                final_prompt,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            
            # Prepare result
            result = {
                "summary": summary,
                "final": answer,
                "prompt": final_prompt,
                "embedding_result": embedding_result,
                "embedding_stats": self.get_embedding_stats(),
                "numbskull_enabled": self.numbskull_pipeline is not None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration with embeddings failed: {e}")
            raise
    
    def run(
        self, 
        user_prompt: str, 
        resource_paths: List[str], 
        inline_resources: List[str]
    ) -> Dict[str, str]:
        """
        Synchronous wrapper for run_with_embeddings
        Maintains compatibility with base class interface
        """
        return asyncio.run(self.run_with_embeddings(user_prompt, resource_paths, inline_resources))
    
    async def run_async(
        self, 
        user_prompt: str, 
        resource_paths: List[str], 
        inline_resources: List[str]
    ) -> Dict[str, str]:
        """Async version using embeddings"""
        return await self.run_with_embeddings(user_prompt, resource_paths, inline_resources)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding performance statistics"""
        stats = self.embedding_stats.copy()
        stats["cache_size"] = len(self.embedding_cache)
        
        if stats["total_embeddings"] > 0:
            stats["avg_embedding_time"] = stats["embedding_time"] / stats["total_embeddings"]
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["total_embeddings"] + stats["cache_hits"])
        else:
            stats["avg_embedding_time"] = 0.0
            stats["cache_hit_rate"] = 0.0
        
        return stats
    
    def clear_embedding_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        if self.numbskull_pipeline:
            self.numbskull_pipeline.clear_cache()
        logger.info("Embedding caches cleared")
    
    async def close(self):
        """Clean up resources"""
        if self.numbskull_pipeline:
            await self.numbskull_pipeline.close()
        logger.info("Numbskull orchestrator closed")


def create_numbskull_orchestrator(
    local_configs: List[Dict[str, Any]],
    remote_config: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None,
    numbskull_config: Optional[Dict[str, Any]] = None
) -> NumbskullDualOrchestrator:
    """
    Factory function to create numbskull-enhanced orchestrator from config dictionaries
    
    Args:
        local_configs: List of local LLM configurations (for LFM2-8B-A1B)
        remote_config: Optional remote LLM configuration (for resource summarization)
        settings: Orchestrator settings
        numbskull_config: Numbskull pipeline configuration
    
    Returns:
        Configured NumbskullDualOrchestrator instance
    """
    # Create local LLM configs
    local_http_configs = [HTTPConfig(**config) for config in local_configs]
    local_llm = LocalLLM(local_http_configs)
    
    # Create resource LLM config
    resource_llm = ResourceLLM(HTTPConfig(**remote_config) if remote_config else None)
    
    # Create settings
    orchestrator_settings = NumbskullOrchestratorSettings(**(settings or {}))
    
    # Create numbskull config if provided
    hybrid_config = None
    if numbskull_config and NUMBSKULL_AVAILABLE:
        hybrid_config = HybridConfig(**numbskull_config)
    
    return NumbskullDualOrchestrator(
        local_llm, 
        resource_llm, 
        orchestrator_settings,
        hybrid_config
    )


def demo_numbskull_orchestrator():
    """Demonstration of the numbskull-enhanced dual LLM orchestrator"""
    
    # Example configurations
    local_configs = [
        {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "LFM2-8B-A1B"
        }
    ]
    
    remote_config = {
        "base_url": "https://api.openai.com",
        "api_key": "your-api-key-here",
        "model": "gpt-4o-mini"
    }
    
    settings = {
        "temperature": 0.7,
        "max_tokens": 512,
        "style": "concise",
        "use_numbskull": True,
        "use_semantic": True,
        "use_mathematical": True,
        "use_fractal": True,
        "fusion_method": "weighted_average",
        "embedding_enhancement": "metadata"
    }
    
    # Create orchestrator
    orchestrator = create_numbskull_orchestrator(
        local_configs, 
        remote_config, 
        settings
    )
    
    # Example usage
    user_prompt = "Analyze the key technical concepts and provide insights"
    resource_paths = ["README.md"]
    inline_resources = ["Additional context: Advanced AI system integration."]
    
    try:
        result = orchestrator.run(user_prompt, resource_paths, inline_resources)
        
        logger.info("✅ Orchestration completed successfully")
        logger.info(f"Summary length: {len(result['summary'])}")
        logger.info(f"Final answer length: {len(result['final'])}")
        logger.info(f"Numbskull enabled: {result['numbskull_enabled']}")
        
        if result.get('embedding_result'):
            logger.info(f"Embedding components: {result['embedding_result']['metadata']['components_used']}")
        
        stats = result.get('embedding_stats', {})
        logger.info(f"Embedding stats: {stats}")
        
        return result
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        return None


if __name__ == "__main__":
    if not NUMBSKULL_AVAILABLE:
        logger.error("Numbskull pipeline not available. Please install numbskull package.")
        sys.exit(1)
    
    demo_numbskull_orchestrator()

