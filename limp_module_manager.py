#!/usr/bin/env python3
"""
LiMp Module Manager: Complete Integration Hub
=============================================

Central management system for all LiMp modules integrated with Numbskull:
- Unified Cognitive Orchestrator
- Enhanced Vector Index
- Enhanced Graph Store  
- Neuro-Symbolic Engine
- Holographic Memory
- TA ULS Transformer
- Signal Processing
- And more...

Provides easy access to all integrated functionality.

Author: Assistant
License: MIT
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModuleStatus:
    """Status of a single module"""
    name: str
    available: bool
    initialized: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiMpModuleManager:
    """
    Central manager for all LiMp + Numbskull integrated modules
    
    Provides unified access to:
    - Cognitive orchestration
    - Vector indexing
    - Graph storage
    - Neuro-symbolic processing
    - Holographic memory
    - And more...
    """
    
    def __init__(self, auto_init: bool = False):
        """
        Initialize module manager
        
        Args:
            auto_init: Automatically initialize all available modules
        """
        self.modules: Dict[str, ModuleStatus] = {}
        self.instances: Dict[str, Any] = {}
        
        logger.info("=" * 70)
        logger.info("LiMp Module Manager Initializing")
        logger.info("=" * 70)
        
        # Discover available modules
        self._discover_modules()
        
        if auto_init:
            asyncio.run(self.initialize_all())
    
    def _discover_modules(self):
        """Discover available modules"""
        
        # Core integrations
        self._check_module("unified_cognitive_orchestrator", "unified_cognitive_orchestrator")
        self._check_module("numbskull_dual_orchestrator", "numbskull_dual_orchestrator")
        self._check_module("enhanced_vector_index", "enhanced_vector_index")
        self._check_module("enhanced_graph_store", "enhanced_graph_store")
        
        # LiMp modules
        self._check_module("neuro_symbolic_engine", "neuro_symbolic_engine")
        self._check_module("holographic_memory", "holographic_memory_system")
        self._check_module("tauls_transformer", "tauls_transformer")
        self._check_module("signal_processing", "signal_processing")
        self._check_module("matrix_processor", "matrix_processor")
        
        # Numbskull
        self._check_module("numbskull", "advanced_embedding_pipeline", 
                          import_path="/home/kill/numbskull")
        
        self._print_discovery_summary()
    
    def _check_module(self, name: str, module_name: str, import_path: Optional[str] = None):
        """Check if a module is available"""
        try:
            import sys
            if import_path and import_path not in sys.path:
                sys.path.insert(0, import_path)
            
            __import__(module_name)
            self.modules[name] = ModuleStatus(
                name=name,
                available=True,
                metadata={"module_name": module_name}
            )
            logger.debug(f"✅ {name} available")
        except Exception as e:
            # Catch all exceptions including SyntaxError
            self.modules[name] = ModuleStatus(
                name=name,
                available=False,
                error=str(e)
            )
            logger.debug(f"❌ {name} not available: {e}")
    
    def _print_discovery_summary(self):
        """Print module discovery summary"""
        available = sum(1 for m in self.modules.values() if m.available)
        total = len(self.modules)
        
        logger.info(f"\nModule Discovery: {available}/{total} available")
        logger.info("-" * 70)
        
        categories = {
            "Core Integration": ["unified_cognitive_orchestrator", "numbskull_dual_orchestrator"],
            "Data Structures": ["enhanced_vector_index", "enhanced_graph_store"],
            "LiMp Modules": ["neuro_symbolic_engine", "holographic_memory", "tauls_transformer", 
                            "signal_processing", "matrix_processor"],
            "Embeddings": ["numbskull"]
        }
        
        for category, module_names in categories.items():
            logger.info(f"\n{category}:")
            for name in module_names:
                if name in self.modules:
                    status = "✅" if self.modules[name].available else "❌"
                    logger.info(f"  {status} {name}")
    
    async def initialize_module(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize a specific module
        
        Args:
            name: Module name
            config: Optional configuration
        
        Returns:
            Success status
        """
        if name not in self.modules:
            logger.error(f"Module {name} not found")
            return False
        
        if not self.modules[name].available:
            logger.error(f"Module {name} not available")
            return False
        
        if self.modules[name].initialized:
            logger.info(f"Module {name} already initialized")
            return True
        
        try:
            logger.info(f"Initializing {name}...")
            
            # Initialize specific modules
            if name == "unified_cognitive_orchestrator":
                from unified_cognitive_orchestrator import UnifiedCognitiveOrchestrator
                self.instances[name] = UnifiedCognitiveOrchestrator(**(config or {}))
            
            elif name == "enhanced_vector_index":
                from enhanced_vector_index import EnhancedVectorIndex
                self.instances[name] = EnhancedVectorIndex(**(config or {}))
            
            elif name == "enhanced_graph_store":
                from enhanced_graph_store import EnhancedGraphStore
                self.instances[name] = EnhancedGraphStore(**(config or {}))
            
            elif name == "numbskull":
                import sys
                sys.path.insert(0, "/home/kill/numbskull")
                from advanced_embedding_pipeline import HybridEmbeddingPipeline, HybridConfig
                cfg = HybridConfig(**(config or {}))
                self.instances[name] = HybridEmbeddingPipeline(cfg)
            
            else:
                logger.warning(f"No initialization handler for {name}")
                return False
            
            self.modules[name].initialized = True
            logger.info(f"✅ {name} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            self.modules[name].error = str(e)
            return False
    
    async def initialize_all(self, config: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize all available modules
        
        Args:
            config: Optional configuration dict keyed by module name
        """
        config = config or {}
        
        for name in self.modules.keys():
            if self.modules[name].available:
                await self.initialize_module(name, config.get(name))
    
    def get_module(self, name: str) -> Optional[Any]:
        """
        Get initialized module instance
        
        Args:
            name: Module name
        
        Returns:
            Module instance or None
        """
        return self.instances.get(name)
    
    def get_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of modules
        
        Args:
            name: Optional specific module name
        
        Returns:
            Status dict
        """
        if name:
            if name in self.modules:
                return {
                    "name": name,
                    "available": self.modules[name].available,
                    "initialized": self.modules[name].initialized,
                    "error": self.modules[name].error
                }
            return {"error": f"Module {name} not found"}
        
        # Return all statuses
        return {
            name: {
                "available": module.available,
                "initialized": module.initialized,
                "error": module.error
            }
            for name, module in self.modules.items()
        }
    
    def get_available_modules(self) -> List[str]:
        """Get list of available modules"""
        return [name for name, module in self.modules.items() if module.available]
    
    def get_initialized_modules(self) -> List[str]:
        """Get list of initialized modules"""
        return [name for name, module in self.modules.items() if module.initialized]
    
    async def close_all(self):
        """Close all initialized modules"""
        logger.info("Closing all modules...")
        
        for name, instance in self.instances.items():
            try:
                if hasattr(instance, 'close'):
                    if asyncio.iscoroutinefunction(instance.close):
                        await instance.close()
                    else:
                        instance.close()
                logger.info(f"✅ Closed {name}")
            except Exception as e:
                logger.warning(f"Error closing {name}: {e}")
        
        self.instances.clear()
        for module in self.modules.values():
            module.initialized = False
    
    def export_status(self, filename: str = "limp_module_status.json"):
        """Export module status to JSON file"""
        status = self.get_status()
        with open(filename, 'w') as f:
            json.dump(status, f, indent=2)
        logger.info(f"✅ Status exported to {filename}")
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("LiMp MODULE MANAGER SUMMARY")
        print("=" * 70)
        
        available = self.get_available_modules()
        initialized = self.get_initialized_modules()
        
        print(f"\nModules Available: {len(available)}/{len(self.modules)}")
        print(f"Modules Initialized: {len(initialized)}/{len(available)}")
        
        print("\n--- AVAILABLE MODULES ---")
        for name in available:
            status = "✅ INIT" if name in initialized else "⭕ READY"
            print(f"  {status} {name}")
        
        print("\n--- UNAVAILABLE MODULES ---")
        unavailable = [name for name, m in self.modules.items() if not m.available]
        for name in unavailable:
            print(f"  ❌ {name}")
            if self.modules[name].error:
                print(f"     Error: {self.modules[name].error[:60]}...")
        
        print("\n" + "=" * 70)


async def demo_module_manager():
    """Demonstration of module manager"""
    print("\n" + "=" * 70)
    print("LiMp MODULE MANAGER DEMO")
    print("=" * 70)
    
    # Create manager
    manager = LiMpModuleManager()
    
    # Show available modules
    manager.print_summary()
    
    # Initialize specific modules
    print("\n--- INITIALIZING MODULES ---")
    
    # Try initializing vector index
    success = await manager.initialize_module("enhanced_vector_index", {
        "embedding_dim": 768,
        "use_numbskull": True,
        "numbskull_config": {"use_fractal": True}
    })
    
    if success:
        print("✅ Vector index ready")
        vector_index = manager.get_module("enhanced_vector_index")
        print(f"   Instance: {type(vector_index).__name__}")
    
    # Try initializing graph store
    success = await manager.initialize_module("enhanced_graph_store", {
        "use_numbskull": True,
        "numbskull_config": {"use_fractal": True}
    })
    
    if success:
        print("✅ Graph store ready")
        graph_store = manager.get_module("enhanced_graph_store")
        print(f"   Instance: {type(graph_store).__name__}")
    
    # Export status
    print("\n--- EXPORTING STATUS ---")
    manager.export_status()
    
    # Final summary
    manager.print_summary()
    
    # Cleanup
    print("\n--- CLEANUP ---")
    await manager.close_all()
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_module_manager())

