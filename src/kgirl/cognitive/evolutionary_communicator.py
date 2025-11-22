#!/usr/bin/env python3
"""
Evolutionary Communicator
=========================
High-level orchestration that combines:
- Dual LLM orchestration (local final inference + remote summarization)
- Neuro-symbolic analysis (MirrorCastEngine + AdaptiveLinkPlanner)
- Digital signal processing and modulation (signal_processing)

This module exposes a thin, runnable facade that composes existing building
blocks in this repo to achieve the requested evolutionary pipeline:
  message → dual-LLM content → neuro-symbolic analysis → adaptive modulation → WAV/IQ output

Author: Assistant
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Local imports from existing modules
import dual_llm_orchestrator as dllo
import neuro_symbolic_engine as nse
import signal_processing as dsp


log = logging.getLogger(__name__)


@dataclass
class EvolutionContext:
    resource_paths: List[str]
    inline_resources: List[str]
    local_llm: Dict[str, Any]  # matches dllo.HTTPConfig fields
    remote_llm: Optional[Dict[str, Any]] = None
    orchestrator_settings: Optional[Dict[str, Any]] = None
    outdir: str = "evolved_output"
    want_wav: bool = True
    want_iq: bool = True
    fec: dsp.FEC = dsp.FEC.HAMMING74
    security: dsp.SecurityConfig = dsp.SecurityConfig()


class EvolutionaryCommunicator:
    """Compose Dual LLM → Neuro-Symbolic → DSP wavecasting."""

    def __init__(self):
        self.mirrorcast = nse.MirrorCastEngine()
        self.planner = nse.AdaptiveLinkPlanner()

    def _build_orchestrator(self, ctx: EvolutionContext) -> dllo.DualLLMOrchestrator:
        local = [ctx.local_llm]
        orch = dllo.create_orchestrator(local, ctx.remote_llm, ctx.orchestrator_settings)
        return orch

    def evolve(self, message: str, ctx: EvolutionContext) -> Dict[str, Any]:
        # 1) Dual-LLM composition
        orchestrator = self._build_orchestrator(ctx)
        result = orchestrator.run(message, ctx.resource_paths, ctx.inline_resources)
        content = result["final"]

        # 2) Neuro-symbolic reflection
        analysis = self.mirrorcast.cast(content)

        # 3) Adaptive RL-driven modulation plan
        mod_cfg, explanation = self.planner.plan(content, analysis)

        # Map chosen modulation to DSP scheme
        scheme_name = mod_cfg.get("modulation", "qpsk").upper()
        try:
            scheme = dsp.ModulationScheme[scheme_name]
        except KeyError:
            scheme = dsp.ModulationScheme.QPSK

        # Build signal configs
        mcfg = dsp.ModConfig(
            sample_rate=mod_cfg.get("sample_rate", 48000),
            symbol_rate=mod_cfg.get("symbol_rate", 1200),
            amplitude=mod_cfg.get("amplitude", 0.7),
            f0=mod_cfg.get("f0", 1200.0),
            f1=mod_cfg.get("f1", 2200.0),
            fc=mod_cfg.get("fc", 1800.0),
            clip=True,
            ofdm_subc=mod_cfg.get("ofdm_subc", 64),
            cp_len=mod_cfg.get("cp_len", 16),
            dsss_chip_rate=mod_cfg.get("dsss_chip_rate", 4800),
        )
        fcfg = dsp.FrameConfig()

        # 4) DSP: encode + modulate + save outputs
        out_paths = dsp.full_process_and_save(
            text=content,
            outdir=Path(ctx.outdir),
            scheme=scheme,
            mcfg=mcfg,
            fcfg=fcfg,
            sec=ctx.security,
            fec_scheme=ctx.fec,
            want_wav=ctx.want_wav,
            want_iq=ctx.want_iq,
            title=f"Evolutionary-{scheme.name}"
        )

        # 5) Feedback for RL (simulate success metric)
        success = True
        self.planner.reward_and_record(
            content,
            mod_cfg,
            explanation,
            success,
            entropy=analysis.get("entropy", 0.0),
            complexity=analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5),
            harmony=analysis.get("love", {}).get("harmony_index", 0.5),
        )

        return {
            "files": {
                "wav": str(out_paths.wav) if out_paths.wav else None,
                "iq": str(out_paths.iq) if out_paths.iq else None,
                "meta": str(out_paths.meta) if out_paths.meta else None,
                "plot": str(out_paths.png) if out_paths.png else None,
            },
            "content_preview": content[:400] + ("..." if len(content) > 400 else ""),
            "orchestrator": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in result.items()},
            "neuro_symbolic": {
                "entropy": analysis.get("entropy"),
                "fractal_dimension": analysis.get("fractal", {}).get("fractal_dimension"),
                "semantic": analysis.get("semantic"),
            },
            "planner": {"explanation": explanation, "config": mod_cfg},
        }


def demo():
    """Minimal demo with a local llama-cpp style endpoint and no remote."""
    ctx = EvolutionContext(
        resource_paths=[],
        inline_resources=["Local-only demo. Provide remote to enable summarization."],
        local_llm={"base_url": "http://127.0.0.1:8080", "mode": "llama-cpp", "model": "local-gguf"},
        remote_llm=None,
        orchestrator_settings={"temperature": 0.7, "max_tokens": 256, "style": "concise"},
    )

    evo = EvolutionaryCommunicator()
    result = evo.evolve("Explain TAU-ULS stability in 3 sentences", ctx)
    print(nse.json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()


