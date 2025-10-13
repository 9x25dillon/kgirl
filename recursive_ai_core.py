#!/usr/bin/env python3
"""
Recursive AI Core - FastAPI Microservice
Implements the recursive cognition API with dynamic depth analysis
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import existing components
from recursive_cognitive_system import RecursiveCognitiveKnowledge
from llm_orchestrator import DualLLMOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class RecursiveCognitionRequest(BaseModel):
    input_text: str
    depth: int = 5
    temperature: float = 0.8
    coherence_threshold: float = 0.6

class RecursiveCognitionResponse(BaseModel):
    depth: int
    insights: List[Dict[str, Any]]
    compiled: Dict[str, Any]
    processing_time: float
    cognitive_state: Dict[str, Any]

class InsightVariation(BaseModel):
    text: str
    type: str
    coherence: float
    sub_analysis: Optional[Dict[str, Any]] = None

@dataclass
class RecursiveProcessor:
    """Core recursive processing engine"""
    max_depth: int = 5
    temperature: float = 0.8
    coherence_threshold: float = 0.6
    
    def __post_init__(self):
        self.cognitive_system = None
        self.llm_orchestrator = None
        self.insights_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the recursive processing components"""
        self.cognitive_system = RecursiveCognitiveKnowledge(
            max_recursion_depth=self.max_depth,
            hallucination_temperature=self.temperature,
            coherence_threshold=self.coherence_threshold
        )
        await self.cognitive_system.initialize()
        
        self.llm_orchestrator = DualLLMOrchestrator()
        logger.info("Recursive processor initialized")
    
    async def hallucinate(self, layer_input: str, context: List[Dict[str, Any]] = None) -> List[InsightVariation]:
        """Generate variations/hallucinations for the current layer"""
        if not self.llm_orchestrator:
            await self.initialize()
            
        variations = []
        
        # Use LLM orchestrator to generate variations
        try:
            prompt = f"Generate 3 creative variations and insights about: {layer_input}"
            result = await self.llm_orchestrator.generate_and_critique(prompt, max_tokens=150)
            
            # Parse the generated content into variations
            candidate = result.get('candidate', '')
            score = result.get('score', 0.5)
            
            # Split into potential variations
            lines = [line.strip() for line in candidate.split('\n') if line.strip()]
            for i, line in enumerate(lines[:3]):
                variation = InsightVariation(
                    text=line,
                    type=f"llm_generated_{i+1}",
                    coherence=min(0.9, score + (i * 0.1))
                )
                variations.append(variation)
                
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, using fallback")
            # Fallback to simple word-based variations
            words = [w.strip('.,!?') for w in layer_input.split() if len(w) > 3]
            if len(words) >= 2:
                variations = [
                    InsightVariation(text=f"{words[0]} enables {words[1]}", type="combination", coherence=0.6),
                    InsightVariation(text=f"{words[1]} requires {words[0]}", type="inverse", coherence=0.5),
                    InsightVariation(text=f"Emergent pattern: {words[0]} + {words[1]}", type="emergence", coherence=0.7)
                ]
        
        return variations
    
    async def analyze_variations(self, variations: List[InsightVariation], depth: int) -> List[Dict[str, Any]]:
        """Analyze variations and extract insights"""
        insights = []
        
        for variation in variations:
            if variation.coherence >= self.coherence_threshold:
                # Use cognitive system for deeper analysis
                if self.cognitive_system:
                    analysis = await self.cognitive_system.recursive_analyze(
                        variation.text, 
                        current_depth=depth,
                        source_query=variation.text
                    )
                    variation.sub_analysis = analysis
                
                insight = {
                    "text": variation.text,
                    "type": variation.type,
                    "coherence": variation.coherence,
                    "depth": depth,
                    "timestamp": time.time(),
                    "sub_analysis": variation.sub_analysis
                }
                insights.append(insight)
        
        return insights
    
    async def compile_insights(self, all_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile all insights into a coherent knowledge structure"""
        if not all_insights:
            return {"compiled_insights": [], "patterns": [], "synthesis": "No insights generated"}
        
        # Extract patterns
        patterns = {}
        for insight in all_insights:
            insight_type = insight.get("type", "unknown")
            patterns[insight_type] = patterns.get(insight_type, 0) + 1
        
        # Find highest coherence insights
        high_coherence = [i for i in all_insights if i.get("coherence", 0) > 0.7]
        
        # Generate synthesis
        synthesis_texts = [i["text"] for i in high_coherence[:3]]
        synthesis = " | ".join(synthesis_texts) if synthesis_texts else "Emergent knowledge synthesis"
        
        return {
            "compiled_insights": all_insights,
            "patterns": patterns,
            "synthesis": synthesis,
            "total_insights": len(all_insights),
            "high_coherence_count": len(high_coherence),
            "depth_distribution": self._analyze_depth_distribution(all_insights)
        }
    
    def _analyze_depth_distribution(self, insights: List[Dict[str, Any]]) -> Dict[int, int]:
        """Analyze the distribution of insights across recursion depths"""
        distribution = {}
        for insight in insights:
            depth = insight.get("depth", 0)
            distribution[depth] = distribution.get(depth, 0) + 1
        return distribution

# FastAPI application
app = FastAPI(title="Recursive AI Core", version="1.0.0")

# Global processor instance
processor = RecursiveProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize the recursive processor on startup"""
    await processor.initialize()

@app.post("/recursive_cognition", response_model=RecursiveCognitionResponse)
async def recursive_cognition(request: RecursiveCognitionRequest):
    """
    Main recursive cognition endpoint
    Triggers dynamic recursion with specified depth
    """
    start_time = time.time()
    
    try:
        # Update processor settings
        processor.max_depth = request.depth
        processor.temperature = request.temperature
        processor.coherence_threshold = request.coherence_threshold
        
        results = []
        layer_input = request.input_text
        
        logger.info(f"Starting recursive analysis with depth={request.depth}")
        
        # Recursive processing loop
        for d in range(request.depth):
            logger.info(f"Processing depth {d}: {layer_input[:50]}...")
            
            # Generate variations for current layer
            variations = await processor.hallucinate(layer_input)
            
            # Analyze variations
            insights = await processor.analyze_variations(variations, d)
            results.extend(insights)
            
            # Update layer input for next iteration
            if insights:
                # Use highest coherence insight as next layer input
                best_insight = max(insights, key=lambda x: x.get("coherence", 0))
                layer_input = best_insight["text"]
            else:
                # Fallback to original input with depth marker
                layer_input = f"[Depth {d+1}] {request.input_text}"
        
        # Compile all insights
        compiled = await processor.compile_insights(results)
        
        # Get cognitive state
        cognitive_state = {}
        if processor.cognitive_system:
            cognitive_state = processor.cognitive_system.get_cognitive_map()
        
        processing_time = time.time() - start_time
        
        return RecursiveCognitionResponse(
            depth=request.depth,
            insights=results,
            compiled=compiled,
            processing_time=processing_time,
            cognitive_state=cognitive_state
        )
        
    except Exception as e:
        logger.error(f"Recursive cognition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/cognitive_state")
async def get_cognitive_state():
    """Get current cognitive state"""
    if processor.cognitive_system:
        return processor.cognitive_system.get_cognitive_map()
    return {"status": "not_initialized"}

@app.post("/reset")
async def reset_processor():
    """Reset the processor state"""
    global processor
    processor = RecursiveProcessor()
    await processor.initialize()
    return {"status": "reset_complete"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)