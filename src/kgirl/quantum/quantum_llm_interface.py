#!/usr/bin/env python3
"""
Quantum LLM Interface
=====================

Conversational interface for the Quantum Holographic Knowledge Database

Provides:
- Natural language interaction with quantum database
- Knowledge synthesis and analysis
- Emergent pattern exploration
- Qualia-based reasoning
- Multi-dimensional knowledge visualization

Author: Assistant
License: MIT
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from quantum_knowledge_database import QuantumHolographicKnowledgeDatabase, NUMBSKULL_AVAILABLE
from quantum_holographic_knowledge_synthesis import KnowledgeQuantum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumLLMInterface:
    """
    Conversational interface for quantum knowledge database
    """

    def __init__(
        self,
        db: QuantumHolographicKnowledgeDatabase,
        llm_orchestrator: Optional[Any] = None
    ):
        self.db = db
        self.llm_orchestrator = llm_orchestrator
        self.conversation_history = []

        logger.info("🤖 Quantum LLM Interface initialized")

    async def process_command(self, command: str) -> str:
        """
        Process a user command

        Commands:
        - ingest <path>: Ingest file or directory
        - query <text>: Query the database
        - analyze <quantum_id>: Analyze a knowledge quantum
        - patterns: Show emergent patterns
        - qualia <quantum_id>: Explore qualia encoding
        - coherence: Show coherence statistics
        - help: Show help
        """

        command = command.strip()
        parts = command.split(maxsplit=1)

        if not parts:
            return "Empty command"

        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        try:
            if cmd == "ingest":
                return await self._handle_ingest(args)
            elif cmd == "query":
                return await self._handle_query(args)
            elif cmd == "analyze":
                return await self._handle_analyze(args)
            elif cmd == "patterns":
                return await self._handle_patterns()
            elif cmd == "qualia":
                return await self._handle_qualia(args)
            elif cmd == "coherence":
                return await self._handle_coherence()
            elif cmd == "stats":
                return await self._handle_stats()
            elif cmd == "help":
                return self._show_help()
            else:
                # Treat as natural language query
                return await self._handle_natural_language(command)

        except Exception as e:
            return f"Error: {e}"

    async def _handle_ingest(self, path: str) -> str:
        """Handle ingestion command"""
        if not path:
            return "Usage: ingest <file_or_directory>"

        path_obj = Path(path)

        if not path_obj.exists():
            return f"Path not found: {path}"

        source_type = "directory" if path_obj.is_dir() else "file"

        try:
            quantum = await self.db.ingest_and_process(path_obj, source_type=source_type)

            response = f"""
✅ Successfully ingested: {path}

Quantum ID: {quantum.quantum_id}
Source Type: {quantum.source_type.value}
Qualia Type: {quantum.qualia_encoding.qualia_type.value}
Coherence: {quantum.coherence_resonance:.3f}
Fractal Completion: {quantum.fractal_completion:.3f}
Emergent Patterns: {len(quantum.emergent_patterns)}
Chaos Attractor: {quantum.chaos_ragged_state.attractor_basin if quantum.chaos_ragged_state else 'N/A'}
Orwellian Layers: {len(quantum.orwells_egged_structure.nested_layers) if quantum.orwells_egged_structure else 'N/A'}
"""
            return response.strip()

        except Exception as e:
            return f"Failed to ingest: {e}"

    async def _handle_query(self, query_text: str) -> str:
        """Handle query command"""
        if not query_text:
            return "Usage: query <search_text>"

        results = await self.db.query_knowledge(query_text, top_k=5)

        if not results:
            return "No results found"

        response = f"🔍 Found {len(results)} results for: '{query_text}'\n\n"

        for i, quantum in enumerate(results, 1):
            response += f"{i}. {quantum.quantum_id}\n"
            response += f"   Source: {quantum.source_path}\n"
            response += f"   Type: {quantum.source_type.value}\n"
            response += f"   Coherence: {quantum.coherence_resonance:.3f}\n"
            response += f"   Qualia: {quantum.qualia_encoding.qualia_type.value}\n"

            if quantum.qualia_encoding:
                response += f"   Consciousness Level: {quantum.qualia_encoding.consciousness_level:.3f}\n"

            response += f"   Content: {quantum.raw_content[:100]}...\n\n"

        return response.strip()

    async def _handle_analyze(self, quantum_id: str) -> str:
        """Handle analyze command"""
        if not quantum_id:
            return "Usage: analyze <quantum_id>"

        quantum = self.db.knowledge_quanta.get(quantum_id)

        if not quantum:
            return f"Quantum not found: {quantum_id}"

        response = f"""
🔬 QUANTUM ANALYSIS: {quantum_id}

=== BASIC INFO ===
Source: {quantum.source_path}
Type: {quantum.source_type.value}
Timestamp: {quantum.timestamp}
Access Count: {quantum.access_count}

=== EMBEDDINGS ===
Hybrid Embedding Dimension: {len(quantum.hybrid_embedding) if quantum.hybrid_embedding is not None else 'N/A'}
Semantic: {'✓' if quantum.semantic_embedding is not None else '✗'}
Mathematical: {'✓' if quantum.mathematical_embedding is not None else '✗'}
Fractal: {'✓' if quantum.fractal_embedding is not None else '✗'}

=== QUANTUM DIMENSIONS ===
Number of Dimensions: {len(quantum.quantum_dimensions)}
"""

        for dim in quantum.quantum_dimensions[:3]:
            response += f"\n  Dimension {dim.dimension_id}:"
            response += f"\n    Coherence: {dim.coherence:.3f}"
            response += f"\n    Entanglement: {dim.entanglement_degree:.3f}"

        if quantum.holographic_encoding:
            response += f"""

=== HOLOGRAPHIC ENCODING ===
Reconstruction Fidelity: {quantum.holographic_encoding.reconstruction_fidelity:.3f}
Fractal Dimension: {quantum.holographic_encoding.fractal_dimension:.3f}
Interference Nodes: {len(quantum.holographic_encoding.interference_nodes)}
"""

        if quantum.qualia_encoding:
            response += f"""

=== QUALIA (SUBJECTIVE EXPERIENCE) ===
Type: {quantum.qualia_encoding.qualia_type.value}
Intentionality: {quantum.qualia_encoding.intentionality}
Consciousness Level: {quantum.qualia_encoding.consciousness_level:.3f}
Phenomenal Properties:
"""
            for key, value in quantum.qualia_encoding.phenomenal_properties.items():
                response += f"  - {key}: {value}\n"

            response += f"Emergent Properties: {', '.join(quantum.qualia_encoding.emergent_properties)}"

        if quantum.chaos_ragged_state:
            response += f"""

=== CHAOS_RAGGED STATE ===
Attractor Basin: {quantum.chaos_ragged_state.attractor_basin}
Chaos Entropy: {quantum.chaos_ragged_state.chaos_entropy:.3f}
Edge of Chaos: {quantum.chaos_ragged_state.edge_of_chaos:.3f}
Bifurcation Points: {len(quantum.chaos_ragged_state.bifurcation_points)}
Ragged Boundaries: {len(quantum.chaos_ragged_state.ragged_boundaries)}
"""

        if quantum.orwells_egged_structure:
            response += f"""

=== ORWELLS-EGGED STRUCTURE ===
Nested Layers: {len(quantum.orwells_egged_structure.nested_layers)}
Surveillance Pattern: {quantum.orwells_egged_structure.surveillance_patterns.get('information_flow', 'N/A')}
Contradictions: {len(quantum.orwells_egged_structure.doublethink_contradictions)}
Thoughtcrimes: {len(quantum.orwells_egged_structure.thoughtcrime_detection)}
"""

        response += f"""

=== RESONANCE & COHERENCE ===
Coherence Resonance: {quantum.coherence_resonance:.3f}
Fractal Completion: {quantum.fractal_completion:.3f}
Knowledge Valence: {quantum.knowledge_valence:.3f}

=== EMERGENT PATTERNS ===
Number of Patterns: {len(quantum.emergent_patterns)}
"""

        for i, pattern in enumerate(quantum.emergent_patterns[:3]):
            response += f"""
  Pattern {i+1}:
    ID: {pattern.pattern_id}
    Type: {pattern.pattern_type}
    Emergence Score: {pattern.emergence_score:.3f}
    Complexity: {pattern.complexity_measure:.3f}
    Coherence: {pattern.coherence_score:.3f}
"""

        return response.strip()

    async def _handle_patterns(self) -> str:
        """Handle patterns command"""
        all_patterns = []

        for quantum in self.db.knowledge_quanta.values():
            all_patterns.extend(quantum.emergent_patterns)

        if not all_patterns:
            return "No emergent patterns detected"

        # Sort by emergence score
        all_patterns.sort(key=lambda p: p.emergence_score, reverse=True)

        response = f"🌟 EMERGENT PATTERNS ({len(all_patterns)} total)\n\n"

        for i, pattern in enumerate(all_patterns[:10], 1):
            response += f"{i}. {pattern.pattern_id}\n"
            response += f"   Type: {pattern.pattern_type}\n"
            response += f"   Emergence Score: {pattern.emergence_score:.3f}\n"
            response += f"   Complexity: {pattern.complexity_measure:.3f}\n"
            response += f"   Coherence: {pattern.coherence_score:.3f}\n"
            response += f"   Properties: {', '.join(pattern.emergent_properties[:3])}\n\n"

        return response.strip()

    async def _handle_qualia(self, quantum_id: str) -> str:
        """Handle qualia exploration command"""
        if not quantum_id:
            return "Usage: qualia <quantum_id>"

        quantum = self.db.knowledge_quanta.get(quantum_id)

        if not quantum or not quantum.qualia_encoding:
            return f"No qualia encoding found for: {quantum_id}"

        qualia = quantum.qualia_encoding

        response = f"""
✨ QUALIA EXPLORATION: {quantum_id}

This knowledge quantum has the following subjective experiential qualities:

QUALIA TYPE: {qualia.qualia_type.value}

INTENTIONALITY (What it's about):
{qualia.intentionality}

PHENOMENAL PROPERTIES (How it feels):
"""

        for key, value in qualia.phenomenal_properties.items():
            description = self._describe_phenomenal_property(key, value)
            response += f"  • {key.upper()}: {description}\n"

        response += f"""

CONSCIOUSNESS LEVEL: {qualia.consciousness_level:.3f}
  (Measure of integrated information / phi)

EMERGENT PHENOMENAL PROPERTIES:
"""

        for prop in qualia.emergent_properties:
            response += f"  • {prop}\n"

        response += f"""

EXPERIENTIAL VECTOR STATISTICS:
  Dimension: {len(qualia.experiential_vector)}
  Norm (Overall intensity): {float(sum(qualia.experiential_vector**2)**0.5):.3f}
  Mean (Central tendency): {float(qualia.experiential_vector.mean()):.3f}
  Std (Variation): {float(qualia.experiential_vector.std()):.3f}

This quantum represents not just information, but a subjective "experience" of that information,
encoded holographically with phenomenal qualities that emerge from its structure.
"""

        return response.strip()

    def _describe_phenomenal_property(self, key: str, value: float) -> str:
        """Describe a phenomenal property in natural language"""
        descriptions = {
            'intensity': f"{value:.2f} - {'Very intense' if value > 5 else 'Moderate' if value > 2 else 'Subtle'}",
            'valence': f"{value:.2f} - {'Positive' if value > 0 else 'Negative' if value < 0 else 'Neutral'}",
            'arousal': f"{value:.2f} - {'High energy' if value > 1 else 'Calm' if value < 0.5 else 'Moderate'}",
            'clarity': f"{value:.2f} - {'Very clear' if value > 0.7 else 'Somewhat clear' if value > 0.4 else 'Vague'}",
            'richness': f"{value:.0f} - {'Very rich' if value > 50 else 'Moderate' if value > 20 else 'Simple'}",
        }
        return descriptions.get(key, f"{value}")

    async def _handle_coherence(self) -> str:
        """Handle coherence statistics command"""
        if not self.db.knowledge_quanta:
            return "No knowledge quanta in database"

        coherences = [q.coherence_resonance for q in self.db.knowledge_quanta.values()]
        completions = [q.fractal_completion for q in self.db.knowledge_quanta.values()]

        import numpy as np
        response = f"""
🌊 COHERENCE & RESONANCE STATISTICS

Total Knowledge Quanta: {len(self.db.knowledge_quanta)}

COHERENCE RESONANCE:
  Mean: {np.mean(coherences):.3f}
  Std Dev: {np.std(coherences):.3f}
  Min: {np.min(coherences):.3f}
  Max: {np.max(coherences):.3f}

FRACTAL COMPLETION:
  Mean: {np.mean(completions):.3f}
  Std Dev: {np.std(completions):.3f}
  Min: {np.min(completions):.3f}
  Max: {np.max(completions):.3f}

HIGH COHERENCE QUANTA (>0.7):
"""

        high_coherence = [q for q in self.db.knowledge_quanta.values() if q.coherence_resonance > 0.7]
        for quantum in high_coherence[:5]:
            response += f"  • {quantum.quantum_id}: {quantum.coherence_resonance:.3f}\n"

        return response.strip()

    async def _handle_stats(self) -> str:
        """Handle statistics command"""
        if not self.db.knowledge_quanta:
            return "No knowledge quanta in database"

        # Gather statistics
        total = len(self.db.knowledge_quanta)
        by_type = {}
        by_qualia = {}

        for quantum in self.db.knowledge_quanta.values():
            source_type = quantum.source_type.value
            by_type[source_type] = by_type.get(source_type, 0) + 1

            if quantum.qualia_encoding:
                qualia_type = quantum.qualia_encoding.qualia_type.value
                by_qualia[qualia_type] = by_qualia.get(qualia_type, 0) + 1

        response = f"""
📊 QUANTUM DATABASE STATISTICS

Total Knowledge Quanta: {total}

BY SOURCE TYPE:
"""
        for stype, count in by_type.items():
            response += f"  • {stype}: {count}\n"

        response += "\nBY QUALIA TYPE:\n"
        for qtype, count in by_qualia.items():
            response += f"  • {qtype}: {count}\n"

        response += f"""

NUMBSKULL STATUS: {'✓ Enabled' if NUMBSKULL_AVAILABLE else '✗ Disabled'}
HOLOGRAPHIC MEMORY: {self.db.holographic_memory.memory_size} slots
CHAOS_RAGGED DIMENSION: {self.db.chaos_ragged.dimension}
QUANTUM PROCESSOR: {self.db.quantum_processor.num_qubits} qubits
"""

        return response.strip()

    async def _handle_natural_language(self, text: str) -> str:
        """Handle natural language input"""
        # For now, treat as a query
        return await self._handle_query(text)

    def _show_help(self) -> str:
        """Show help message"""
        return """
🌌 QUANTUM LLM INTERFACE - COMMANDS

INGESTION:
  ingest <path>          - Ingest file or directory into quantum database

QUERYING:
  query <text>           - Search for knowledge quanta
  <natural language>     - Direct natural language query

ANALYSIS:
  analyze <quantum_id>   - Deep analysis of a knowledge quantum
  qualia <quantum_id>    - Explore subjective experiential qualities
  patterns               - Show emergent patterns across all quanta

STATISTICS:
  stats                  - Show database statistics
  coherence              - Show coherence and resonance statistics

HELP:
  help                   - Show this help message

FEATURES:
  • Multi-source ingestion (PDFs, .py, text, directories)
  • Hybrid embeddings (semantic, mathematical, fractal via Numbskull)
  • Quantum-dimensional encoding
  • Holographic storage with fractal encoding
  • Chaos_Ragged emergent pattern detection
  • Orwells-egged hierarchical structuring
  • Qualia encoding (subjective experiential knowledge)
  • Coherence resonance completion
"""

    async def run_interactive(self):
        """Run interactive REPL"""
        print("\n" + "="*80)
        print("🌌 QUANTUM HOLOGRAPHIC KNOWLEDGE SYNTHESIS - LLM INTERFACE 🌌")
        print("="*80)
        print("Type 'help' for commands, 'quit' to exit\n")

        while True:
            try:
                user_input = input("quantum> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n✨ Closing quantum database...")
                    await self.db.close()
                    print("Goodbye!\n")
                    break

                response = await self.process_command(user_input)
                print(f"\n{response}\n")

            except KeyboardInterrupt:
                print("\n\n✨ Interrupted. Closing...")
                await self.db.close()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Quantum Holographic Knowledge Synthesis - LLM Interface"
    )
    parser.add_argument(
        '--db',
        default='quantum_knowledge.db',
        help='Database file path'
    )
    parser.add_argument(
        '--no-numbskull',
        action='store_true',
        help='Disable Numbskull embeddings'
    )
    parser.add_argument(
        '--command',
        help='Run a single command and exit'
    )

    args = parser.parse_args()

    # Create database
    db = QuantumHolographicKnowledgeDatabase(
        db_path=args.db,
        enable_numbskull=not args.no_numbskull and NUMBSKULL_AVAILABLE
    )

    # Create interface
    interface = QuantumLLMInterface(db)

    if args.command:
        # Run single command
        response = await interface.process_command(args.command)
        print(response)
        await db.close()
    else:
        # Run interactive mode
        await interface.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
