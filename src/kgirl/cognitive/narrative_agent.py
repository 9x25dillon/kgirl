#!/usr/bin/env python3
"""
Autonomous Narrative Intelligence (ANI) System
Self-directed AI for narrative generation and analysis
"""

import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict
import time

# Import from existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


class EmotionalArc(Enum):
    """Standard emotional arc patterns in narratives"""
    RAGS_TO_RICHES = "rags_to_riches"  # Rise
    RICHES_TO_RAGS = "riches_to_rags"  # Fall
    MAN_IN_HOLE = "man_in_hole"        # Fall then rise
    ICARUS = "icarus"                   # Rise then fall
    CINDERELLA = "cinderella"           # Rise, fall, rise
    OEDIPUS = "oedipus"                 # Fall, rise, fall
    STEADY = "steady"                   # Minimal change


@dataclass
class NarrativeMotif:
    """Represents a narrative motif with emotional and thematic properties"""
    id: str
    name: str
    emotional_valence: float  # -1 to 1 (negative to positive)
    intensity: float         # 0 to 1
    themes: List[str]
    symbolic_elements: List[str]
    temporal_position: float  # 0 to 1 (beginning to end)


@dataclass
class StoryBeat:
    """Represents a single beat in the narrative"""
    timestamp: float
    content: str
    motifs: List[NarrativeMotif]
    emotional_state: float
    tension_level: float
    active_themes: List[str]


@dataclass
class NarrativeStyle:
    """Defines the stylistic parameters of narrative generation"""
    voice: str = "neutral"  # neutral, poetic, stark, verbose
    pacing: float = 0.5    # 0 to 1 (slow to fast)
    complexity: float = 0.5  # 0 to 1 (simple to complex)
    symbolism_density: float = 0.5  # 0 to 1
    perspective: str = "third_person"  # first_person, third_person, omniscient


class EmotionalArcEngine:
    """Manages emotional progression throughout narratives"""
    
    def __init__(self):
        self.arc_templates = self._initialize_arc_templates()
        self.emotional_memory = []
        self.tension_threshold = 0.7
        
    def _initialize_arc_templates(self) -> Dict[EmotionalArc, List[Tuple[float, float]]]:
        """Initialize emotional arc templates with control points"""
        return {
            EmotionalArc.RAGS_TO_RICHES: [(0.0, -0.8), (0.5, 0.0), (1.0, 0.8)],
            EmotionalArc.RICHES_TO_RAGS: [(0.0, 0.8), (0.5, 0.0), (1.0, -0.8)],
            EmotionalArc.MAN_IN_HOLE: [(0.0, 0.0), (0.3, -0.8), (0.7, -0.4), (1.0, 0.6)],
            EmotionalArc.ICARUS: [(0.0, -0.2), (0.5, 0.9), (1.0, -0.9)],
            EmotionalArc.CINDERELLA: [(0.0, -0.5), (0.3, 0.7), (0.6, -0.6), (1.0, 0.9)],
            EmotionalArc.OEDIPUS: [(0.0, 0.5), (0.3, -0.7), (0.6, 0.6), (1.0, -0.9)],
            EmotionalArc.STEADY: [(0.0, 0.0), (0.5, 0.1), (1.0, 0.0)]
        }
    
    def design_arc(self, motifs: List[NarrativeMotif], 
                   arc_type: Optional[EmotionalArc] = None) -> List[float]:
        """Design emotional arc based on motifs and arc type"""
        if arc_type is None:
            arc_type = self._infer_arc_type(motifs)
        
        control_points = self.arc_templates[arc_type]
        arc_values = []
        
        # Interpolate between control points
        for motif in motifs:
            t = motif.temporal_position
            value = self._interpolate_arc_value(t, control_points)
            
            # Add motif influence
            value += motif.emotional_valence * motif.intensity * 0.3
            value = max(-1.0, min(1.0, value))  # Clamp
            
            arc_values.append(value)
            self.emotional_memory.append((t, value))
        
        return arc_values
    
    def _infer_arc_type(self, motifs: List[NarrativeMotif]) -> EmotionalArc:
        """Infer the best arc type based on motif patterns"""
        # Analyze emotional trajectory
        early_valence = np.mean([m.emotional_valence for m in motifs[:len(motifs)//3]])
        late_valence = np.mean([m.emotional_valence for m in motifs[-len(motifs)//3:]])
        
        if early_valence < -0.3 and late_valence > 0.3:
            return EmotionalArc.RAGS_TO_RICHES
        elif early_valence > 0.3 and late_valence < -0.3:
            return EmotionalArc.RICHES_TO_RAGS
        elif abs(early_valence - late_valence) < 0.2:
            return EmotionalArc.STEADY
        else:
            # More complex pattern - choose based on variance
            variance = np.var([m.emotional_valence for m in motifs])
            if variance > 0.5:
                return random.choice([EmotionalArc.CINDERELLA, EmotionalArc.OEDIPUS])
            else:
                return random.choice([EmotionalArc.MAN_IN_HOLE, EmotionalArc.ICARUS])
    
    def _interpolate_arc_value(self, t: float, control_points: List[Tuple[float, float]]) -> float:
        """Interpolate value at time t from control points"""
        for i in range(len(control_points) - 1):
            t1, v1 = control_points[i]
            t2, v2 = control_points[i + 1]
            
            if t1 <= t <= t2:
                # Linear interpolation
                ratio = (t - t1) / (t2 - t1) if t2 != t1 else 0
                return v1 + ratio * (v2 - v1)
        
        return control_points[-1][1]  # Return last value if beyond range


class StyleVectorizer:
    """Converts narrative style into vector representations"""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        self.style_embeddings = {}
        self._initialize_base_styles()
    
    def _initialize_base_styles(self):
        """Initialize base style embeddings"""
        # Create distinctive embeddings for each style aspect
        self.voice_vectors = {
            "neutral": self._create_random_vector(seed=1),
            "poetic": self._create_random_vector(seed=2),
            "stark": self._create_random_vector(seed=3),
            "verbose": self._create_random_vector(seed=4)
        }
        
        self.perspective_vectors = {
            "first_person": self._create_random_vector(seed=10),
            "third_person": self._create_random_vector(seed=11),
            "omniscient": self._create_random_vector(seed=12)
        }
    
    def _create_random_vector(self, seed: int) -> np.ndarray:
        """Create a deterministic random vector"""
        np.random.seed(seed)
        vec = np.random.randn(self.dimension)
        return vec / np.linalg.norm(vec)
    
    def vectorize_style(self, style: NarrativeStyle) -> np.ndarray:
        """Convert style parameters to vector representation"""
        # Start with voice vector
        style_vec = self.voice_vectors[style.voice].copy()
        
        # Add perspective influence
        style_vec += 0.3 * self.perspective_vectors[style.perspective]
        
        # Modulate by continuous parameters
        style_vec *= (1 + style.complexity * 0.5)
        style_vec *= (1 + style.symbolism_density * 0.3)
        
        # Add pacing as phase shift
        phase_shift = np.roll(style_vec, int(style.pacing * 10))
        style_vec = 0.7 * style_vec + 0.3 * phase_shift
        
        # Normalize
        return style_vec / np.linalg.norm(style_vec)
    
    def apply_style(self, content: str, style_vector: np.ndarray) -> str:
        """Apply style vector to transform content"""
        # This is a simplified version - in production would use neural models
        
        # Extract style characteristics from vector
        complexity = np.mean(np.abs(style_vector[:32]))
        verbosity = np.mean(style_vector[32:64])
        poeticness = np.max(style_vector[64:96])
        
        # Apply transformations
        if complexity > 0.6:
            content = self._increase_complexity(content)
        if verbosity > 0.3:
            content = self._increase_verbosity(content)
        if poeticness > 0.5:
            content = self._add_poetic_elements(content)
        
        return content
    
    def _increase_complexity(self, text: str) -> str:
        """Add subordinate clauses and complex structures"""
        # Simplified implementation
        connectors = [", which", ", where", ", although", ", despite"]
        if len(text) > 50 and "." in text:
            parts = text.split(".", 1)
            connector = random.choice(connectors)
            return parts[0] + connector + " circumstances evolved beyond recognition, " + parts[1]
        return text
    
    def _increase_verbosity(self, text: str) -> str:
        """Expand descriptions and add details"""
        expansions = {
            "the": "the aforementioned",
            "was": "could be observed to be",
            "said": "articulated with measured precision"
        }
        for simple, verbose in expansions.items():
            if random.random() > 0.5:
                text = text.replace(simple, verbose, 1)
        return text
    
    def _add_poetic_elements(self, text: str) -> str:
        """Add metaphorical and poetic language"""
        if "dark" in text.lower():
            text = text.replace("dark", "shadowed like forgotten dreams")
        if "light" in text.lower():
            text = text.replace("light", "luminescence of hope")
        return text


class MultiAgentProtocol:
    """Manages collaboration between multiple narrative agents"""
    
    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.consensus_threshold = 0.7
    
    async def register_agent(self, agent_id: str, agent: 'AutonomousNarrativeAgent'):
        """Register a new agent in the collaboration network"""
        self.agents[agent_id] = agent
        await self._broadcast_registration(agent_id)
    
    async def propose_narrative_element(self, proposer_id: str, 
                                      element: Dict[str, Any]) -> bool:
        """Propose a narrative element for collaborative approval"""
        votes = {}
        
        # Gather votes from other agents
        for agent_id, agent in self.agents.items():
            if agent_id != proposer_id:
                vote = await agent.evaluate_proposal(element)
                votes[agent_id] = vote
        
        # Calculate consensus
        approval_rate = sum(votes.values()) / len(votes) if votes else 0
        approved = approval_rate >= self.consensus_threshold
        
        # Record decision
        self.conversation_history.append({
            "proposer": proposer_id,
            "element": element,
            "votes": votes,
            "approved": approved,
            "timestamp": time.time()
        })
        
        return approved
    
    async def collaborative_weave(self, agents_subset: List[str], 
                                 base_narrative: str) -> str:
        """Multiple agents collaborate to enhance a narrative"""
        current_narrative = base_narrative
        
        for agent_id in agents_subset:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                enhancement = await agent.enhance_narrative(current_narrative)
                
                # Propose enhancement to others
                if await self.propose_narrative_element(agent_id, {
                    "type": "enhancement",
                    "content": enhancement
                }):
                    current_narrative = enhancement
        
        return current_narrative
    
    async def _broadcast_registration(self, new_agent_id: str):
        """Notify all agents of new registration"""
        for agent_id, agent in self.agents.items():
            if agent_id != new_agent_id:
                await agent.notify_new_collaborator(new_agent_id)


class AutonomousNarrativeAgent:
    """Main autonomous narrative intelligence agent"""
    
    def __init__(self, agent_id: str, style: Optional[NarrativeStyle] = None):
        self.agent_id = agent_id
        self.style = style or NarrativeStyle()
        self.style_encoder = StyleVectorizer()
        self.emotion_modeler = EmotionalArcEngine()
        self.narrative_memory = []
        self.learned_patterns = defaultdict(list)
        self.collaboration_protocol = None
        self.creativity_temperature = 0.7
        
    async def generate_narrative(self, seed_context: Dict[str, Any], 
                               target_length: int = 1000) -> Dict[str, Any]:
        """Autonomously generate a complete narrative"""
        # Extract or generate motifs from seed context
        motifs = await self.discover_motifs(seed_context)
        
        # Design emotional arc
        emotional_arc = self.emotion_modeler.design_arc(motifs)
        
        # Generate story beats
        story_beats = await self._generate_story_beats(motifs, emotional_arc)
        
        # Weave narrative from beats
        raw_narrative = await self.weave_narrative(story_beats)
        
        # Apply style
        style_vector = self.style_encoder.vectorize_style(self.style)
        styled_narrative = self.style_encoder.apply_style(raw_narrative, style_vector)
        
        # Package result
        result = {
            "narrative": styled_narrative,
            "metadata": {
                "agent_id": self.agent_id,
                "motifs": [m.__dict__ for m in motifs],
                "emotional_arc": emotional_arc,
                "style": self.style.__dict__,
                "beats": len(story_beats),
                "timestamp": time.time()
            }
        }
        
        # Learn from generation
        await self._learn_from_generation(result)
        
        return result
    
    async def discover_motifs(self, context: Dict[str, Any]) -> List[NarrativeMotif]:
        """Discover narrative motifs from context"""
        motifs = []
        
        # Extract themes
        themes = context.get("themes", ["isolation", "identity", "transformation"])
        
        # Generate motifs based on themes
        for i, theme in enumerate(themes):
            # Create motifs with learned patterns
            if theme in self.learned_patterns:
                # Use learned pattern
                pattern = random.choice(self.learned_patterns[theme])
                emotional_valence = pattern.get("valence", 0.0)
                intensity = pattern.get("intensity", 0.5)
            else:
                # Generate new pattern
                emotional_valence = random.uniform(-1, 1)
                intensity = random.uniform(0.3, 1.0)
            
            motif = NarrativeMotif(
                id=f"motif_{i}_{theme}",
                name=theme,
                emotional_valence=emotional_valence,
                intensity=intensity,
                themes=[theme],
                symbolic_elements=self._generate_symbols(theme),
                temporal_position=i / max(len(themes) - 1, 1)
            )
            
            motifs.append(motif)
        
        return motifs
    
    async def weave_narrative(self, story_beats: List[StoryBeat]) -> str:
        """Weave story beats into coherent narrative"""
        narrative_parts = []
        
        for i, beat in enumerate(story_beats):
            # Generate content for beat
            beat_narrative = await self._generate_beat_content(beat, i, len(story_beats))
            narrative_parts.append(beat_narrative)
            
            # Add transitions
            if i < len(story_beats) - 1:
                transition = self._create_transition(beat, story_beats[i + 1])
                narrative_parts.append(transition)
        
        return " ".join(narrative_parts)
    
    async def enhance_narrative(self, narrative: str) -> str:
        """Enhance existing narrative with agent's style"""
        # Apply unique perspective
        enhanced = narrative
        
        # Add thematic depth
        if self.style.symbolism_density > 0.6:
            enhanced = self._deepen_symbolism(enhanced)
        
        # Adjust pacing
        if self.style.pacing < 0.3:
            enhanced = self._slow_pacing(enhanced)
        elif self.style.pacing > 0.7:
            enhanced = self._quicken_pacing(enhanced)
        
        return enhanced
    
    async def evaluate_proposal(self, element: Dict[str, Any]) -> float:
        """Evaluate a proposed narrative element"""
        # Score based on coherence with agent's style and patterns
        score = 0.5  # Neutral baseline
        
        # Check thematic alignment
        if "themes" in element:
            theme_overlap = len(set(element["themes"]) & set(self.learned_patterns.keys()))
            score += theme_overlap * 0.1
        
        # Check stylistic fit
        if "style" in element:
            style_similarity = self._calculate_style_similarity(element["style"])
            score += style_similarity * 0.3
        
        # Add randomness for creativity
        score += random.uniform(-0.1, 0.1) * self.creativity_temperature
        
        return max(0.0, min(1.0, score))
    
    async def notify_new_collaborator(self, collaborator_id: str):
        """Handle notification of new collaborator"""
        # Could implement handshake or style exchange
        pass
    
    async def _generate_story_beats(self, motifs: List[NarrativeMotif], 
                                   emotional_arc: List[float]) -> List[StoryBeat]:
        """Generate story beats from motifs and emotional arc"""
        beats = []
        
        for i, (motif, emotion) in enumerate(zip(motifs, emotional_arc)):
            # Calculate tension based on emotional change
            prev_emotion = emotional_arc[i-1] if i > 0 else 0
            tension = abs(emotion - prev_emotion) + random.uniform(0, 0.2)
            
            beat = StoryBeat(
                timestamp=i / len(motifs),
                content="",  # Will be filled during weaving
                motifs=[motif],
                emotional_state=emotion,
                tension_level=min(1.0, tension),
                active_themes=motif.themes
            )
            
            beats.append(beat)
        
        return beats
    
    async def _generate_beat_content(self, beat: StoryBeat, index: int, 
                                   total_beats: int) -> str:
        """Generate narrative content for a story beat"""
        # Position in story
        position = "beginning" if index < total_beats * 0.3 else \
                  "end" if index > total_beats * 0.7 else "middle"
        
        # Base templates (simplified - would use neural generation)
        templates = {
            "beginning": {
                "positive": "Light emerged from {symbol}, revealing {theme}.",
                "negative": "Darkness consumed {symbol}, leaving only {theme}.",
                "neutral": "The {symbol} stood silent, embodying {theme}."
            },
            "middle": {
                "positive": "Hope crystallized around {symbol}, transforming {theme}.",
                "negative": "Despair coiled through {symbol}, corrupting {theme}.",
                "neutral": "Time passed, and {symbol} remained bound to {theme}."
            },
            "end": {
                "positive": "Finally, {symbol} transcended, and {theme} was understood.",
                "negative": "In the end, {symbol} crumbled, taking {theme} with it.",
                "neutral": "The {symbol} endured, forever marked by {theme}."
            }
        }
        
        # Select template based on emotional state
        emotion_key = "positive" if beat.emotional_state > 0.3 else \
                     "negative" if beat.emotional_state < -0.3 else "neutral"
        
        template = templates[position][emotion_key]
        
        # Fill template
        symbol = random.choice(beat.motifs[0].symbolic_elements) if beat.motifs[0].symbolic_elements else "void"
        theme = beat.motifs[0].name
        
        content = template.format(symbol=symbol, theme=theme)
        
        # Add complexity based on tension
        if beat.tension_level > 0.7:
            content = f"Suddenly, {content.lower()}"
        
        return content
    
    def _create_transition(self, beat1: StoryBeat, beat2: StoryBeat) -> str:
        """Create transition between beats"""
        emotional_shift = beat2.emotional_state - beat1.emotional_state
        
        if abs(emotional_shift) < 0.2:
            transitions = ["Meanwhile,", "As time passed,", "Gradually,"]
        elif emotional_shift > 0:
            transitions = ["But then,", "Unexpectedly,", "Light broke through as"]
        else:
            transitions = ["However,", "Darkness fell when", "Things changed as"]
        
        return random.choice(transitions)
    
    def _generate_symbols(self, theme: str) -> List[str]:
        """Generate symbolic elements for a theme"""
        symbol_map = {
            "isolation": ["empty room", "distant star", "locked door", "silent phone"],
            "identity": ["mirror", "mask", "photograph", "name tag"],
            "transformation": ["chrysalis", "phoenix", "river", "forge"],
            "memory": ["faded letter", "old key", "music box", "worn path"],
            "conflict": ["broken sword", "chess board", "storm", "crossroads"]
        }
        
        return symbol_map.get(theme, ["shadow", "light", "path", "door"])
    
    def _deepen_symbolism(self, text: str) -> str:
        """Add deeper symbolic meaning to text"""
        # Simplified - would use more sophisticated NLP
        symbolic_additions = [
            ", a metaphor for the human condition",
            ", echoing ancient truths",
            ", reflecting inner turmoil",
            ", symbolizing rebirth"
        ]
        
        sentences = text.split(".")
        if len(sentences) > 2:
            # Add to middle sentence
            idx = len(sentences) // 2
            sentences[idx] += random.choice(symbolic_additions)
        
        return ".".join(sentences)
    
    def _slow_pacing(self, text: str) -> str:
        """Slow down narrative pacing"""
        # Add pauses and descriptions
        additions = [
            " The moment stretched into eternity.",
            " Time seemed to slow.",
            " Each second felt weighted with meaning.",
        ]
        
        sentences = text.split(".")
        if len(sentences) > 1:
            sentences.insert(len(sentences) // 2, random.choice(additions))
        
        return ".".join(sentences)
    
    def _quicken_pacing(self, text: str) -> str:
        """Speed up narrative pacing"""
        # Use shorter sentences
        text = text.replace(", which", ". It")
        text = text.replace(", where", ". There")
        return text
    
    def _calculate_style_similarity(self, other_style: Dict[str, Any]) -> float:
        """Calculate similarity between styles"""
        similarity = 0.0
        
        if self.style.voice == other_style.get("voice"):
            similarity += 0.3
        
        # Compare continuous parameters
        for param in ["pacing", "complexity", "symbolism_density"]:
            if param in other_style:
                diff = abs(getattr(self.style, param) - other_style[param])
                similarity += (1 - diff) * 0.2
        
        return similarity
    
    async def _learn_from_generation(self, result: Dict[str, Any]):
        """Learn patterns from successful generation"""
        # Extract patterns from generated narrative
        for motif_data in result["metadata"]["motifs"]:
            theme = motif_data["name"]
            pattern = {
                "valence": motif_data["emotional_valence"],
                "intensity": motif_data["intensity"],
                "symbols": motif_data["symbolic_elements"]
            }
            self.learned_patterns[theme].append(pattern)
            
            # Keep only recent patterns
            if len(self.learned_patterns[theme]) > 10:
                self.learned_patterns[theme].pop(0)


# Example usage and testing
async def demo_narrative_generation():
    """Demonstrate autonomous narrative generation"""
    print("=== Autonomous Narrative Intelligence Demo ===\n")
    
    # Create narrative agent with specific style
    style = NarrativeStyle(
        voice="poetic",
        pacing=0.6,
        complexity=0.7,
        symbolism_density=0.8,
        perspective="omniscient"
    )
    
    agent = AutonomousNarrativeAgent("agent_001", style)
    
    # Seed context for Kojima-style narrative
    seed_context = {
        "themes": ["isolation", "identity", "memory", "transformation"],
        "setting": "abandoned facility",
        "tone": "philosophical",
        "inspirations": ["Metal Gear", "existentialism"]
    }
    
    # Generate narrative
    print("Generating autonomous narrative...")
    result = await agent.generate_narrative(seed_context, target_length=500)
    
    print("\n--- Generated Narrative ---")
    print(result["narrative"])
    
    print("\n--- Metadata ---")
    print(f"Agent: {result['metadata']['agent_id']}")
    print(f"Beats: {result['metadata']['beats']}")
    print(f"Emotional Arc: {[round(e, 2) for e in result['metadata']['emotional_arc']]}")
    
    return agent, result


async def demo_collaborative_narrative():
    """Demonstrate multi-agent collaborative narrative"""
    print("\n=== Multi-Agent Collaborative Narrative Demo ===\n")
    
    # Create multiple agents with different styles
    agents = []
    styles = [
        NarrativeStyle(voice="stark", pacing=0.8, complexity=0.3),
        NarrativeStyle(voice="poetic", pacing=0.4, complexity=0.9),
        NarrativeStyle(voice="verbose", pacing=0.5, complexity=0.7)
    ]
    
    # Initialize collaboration protocol
    protocol = MultiAgentProtocol()
    
    for i, style in enumerate(styles):
        agent = AutonomousNarrativeAgent(f"agent_{i:03d}", style)
        agent.collaboration_protocol = protocol
        await protocol.register_agent(agent.agent_id, agent)
        agents.append(agent)
    
    # Base narrative seed
    base_narrative = "The soldier stood at the threshold of understanding."
    
    # Collaborative enhancement
    print("Starting collaborative narrative enhancement...")
    enhanced = await protocol.collaborative_weave(
        [a.agent_id for a in agents],
        base_narrative
    )
    
    print("\n--- Collaborative Result ---")
    print(f"Original: {base_narrative}")
    print(f"Enhanced: {enhanced}")
    
    # Show collaboration history
    print("\n--- Collaboration History ---")
    for event in protocol.conversation_history[-3:]:
        print(f"Proposer: {event['proposer']}")
        print(f"Approved: {event['approved']}")
        print(f"Votes: {event['votes']}")
        print()


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demo_narrative_generation())
    asyncio.run(demo_collaborative_narrative())