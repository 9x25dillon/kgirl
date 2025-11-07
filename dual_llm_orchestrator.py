#!/usr/bin/env python3
"""
Dual LLM Orchestration System
=============================

This module implements a sophisticated dual LLM system where:
- Local LLM handles final inference and decision making
- Remote LLM provides resource-only summarization and structuring
- Orchestrator coordinates between the two systems

Author: Assistant  
License: MIT
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"  # ["openai-chat","openai-completions","llama-cpp","textgen-webui"]
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: 
        raise NotImplementedError

class LocalLLM(BaseLLM):
    """Local LLM for final inference and decision making"""
    
    def __init__(self, configs: List[HTTPConfig]):
        if not HAS_REQUESTS:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last_error = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                return self._call(cfg, prompt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Local LLM config {self.idx} failed: {e}")
                self.idx = (self.idx + 1) % len(self.configs)
        
        raise last_error or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        session = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                response = session.post(
                    url, headers=headers, json=body, 
                    timeout=cfg.timeout, verify=cfg.verify_ssl
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < cfg.max_retries - 1:
                    time.sleep(cfg.retry_delay * (2 ** attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: 
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
            
        elif mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: 
                headers["Authorization"] = f"Bearer {cfg.api_key}"
            
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
            
        elif mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {
                "prompt": prompt, 
                "temperature": kwargs.get("temperature", 0.7), 
                "n_predict": kwargs.get("max_tokens", 512)
            }
            data = self._post(cfg, url, {}, body)
            
            if "content" in data: 
                return data["content"]
            if "choices" in data and data["choices"]: 
                return data["choices"][0].get("text", "")
            return data.get("text", "")
            
        elif mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {
                "prompt": prompt, 
                "max_new_tokens": kwargs.get("max_tokens", 512), 
                "temperature": kwargs.get("temperature", 0.7)
            }
            data = self._post(cfg, url, {}, body)
            return data.get("results", [{}])[0].get("text", "")
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    """Remote LLM constrained to resource-only summarization"""
    
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        # Constrained to resources-only summarization
        if self.cfg is None or not HAS_REQUESTS:
            return LocalSummarizer().summarize(prompt)
            
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key: 
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
            
        system_prompt = (
            "You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
            "Do not add external knowledge or make inferences beyond what is explicitly stated."
        )
        
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        
        session = requests.Session()
        response = session.post(
            url, headers=headers, json=body, 
            timeout=self.cfg.timeout, verify=self.cfg.verify_ssl
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    """Fallback local summarizer when remote LLM is unavailable"""
    
    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", 
            "did", "will", "would", "could", "should", "from", "that", "this", "it", "as"
        }
    
    def summarize(self, text: str) -> str:
        text = " ".join(text.split())
        if not text: 
            return "No content to summarize."
            
        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        if not sentences: 
            return text[:300] + ("..." if len(text) > 300 else "")
        
        # Score sentences by length + term frequency (simple heuristic)
        words = [w.lower().strip(",;:()[]") for w in text.split()]
        freq: Dict[str, int] = {}
        for word in words:
            if word and word not in self.stop_words: 
                freq[word] = freq.get(word, 0) + 1
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = [w.lower().strip(",;:()[]") for w in sentence.split()]
            score = len(sentence) * 0.1 + sum(freq.get(w, 0) for w in sentence_words)
            scored_sentences.append((sentence, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s, _ in scored_sentences[:min(6, len(scored_sentences))]]
        keep.sort(key=lambda k: sentences.index(k))
        
        result = " ".join(keep)
        return result[:800] + ("..." if len(result) > 800 else "")

class DualLLMOrchestrator:
    """Orchestrates coordination between local and resource LLMs"""
    
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local = local
        self.resource = resource
        self.settings = settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        """Load and combine resources from files and inline text"""
        parts = []
        
        # Load from files
        for path_str in paths:
            path = Path(path_str)
            if path.exists() and path.is_file():
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    parts.append(content)
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
                    parts.append(f"[[UNREADABLE_FILE:{path.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{path_str}]]")
        
        # Add inline resources
        parts.extend([str(x) for x in inline])
        
        # Combine and truncate
        blob = "\n\n".join(parts)
        return blob[:self.settings.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str, str]:
        """Compose the final prompt using resource summarization"""
        # Load and summarize resources
        resource_text = self._load_resources(resource_paths, inline_resources)
        
        resource_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{resource_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, 
            max_tokens=self.settings.max_tokens
        )
        
        # Create final prompt for local LLM
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{resource_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.settings.style}. Be clear and directly actionable."
        )
        
        return final_prompt, resource_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str, str]:
        """Execute the full dual LLM orchestration"""
        final_prompt, summary = self.compose(user_prompt, resource_paths, inline_resources)
        
        answer = self.local.generate(
            final_prompt, 
            temperature=self.settings.temperature, 
            max_tokens=self.settings.max_tokens
        )
        
        return {
            "summary": summary, 
            "final": answer, 
            "prompt": final_prompt
        }

    async def run_async(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str, str]:
        """Async version for better performance"""
        # For now, just wrap the sync version
        # In a full implementation, this would use async HTTP clients
        return self.run(user_prompt, resource_paths, inline_resources)

def create_orchestrator(
    local_configs: List[Dict[str, Any]], 
    remote_config: Optional[Dict[str, Any]] = None,
    settings: Optional[Dict[str, Any]] = None
) -> DualLLMOrchestrator:
    """Factory function to create orchestrator from config dictionaries"""
    
    # Create local LLM configs
    local_http_configs = [HTTPConfig(**config) for config in local_configs]
    local_llm = LocalLLM(local_http_configs)
    
    # Create resource LLM config
    resource_llm = ResourceLLM(HTTPConfig(**remote_config) if remote_config else None)
    
    # Create settings
    orchestrator_settings = OrchestratorSettings(**(settings or {}))
    
    return DualLLMOrchestrator(local_llm, resource_llm, orchestrator_settings)

def demo_orchestrator():
    """Demonstration of the dual LLM orchestrator"""
    
    # Example configurations
    local_configs = [
        {
            "base_url": "http://127.0.0.1:8080",
            "mode": "llama-cpp",
            "model": "local-gguf"
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
        "style": "concise"
    }
    
    # Create orchestrator
    orchestrator = create_orchestrator(local_configs, remote_config, settings)
    
    # Example usage
    user_prompt = "Create a 2-paragraph technical summary"
    resource_paths = ["example_document.txt"]
    inline_resources = ["Additional context: This is about AI systems."]
    
    try:
        result = orchestrator.run(user_prompt, resource_paths, inline_resources)
        
        logger.info("Orchestration completed successfully")
        logger.info(f"Summary length: {len(result['summary'])}")
        logger.info(f"Final answer length: {len(result['final'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        return None

if __name__ == "__main__":
    demo_orchestrator()