# NewThought Usage Examples

## Example 1: Basic Thought Generation

```python
import httpx
import asyncio

async def basic_generation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/newthought/generate",
            json={
                "seed_text": "Quantum entanglement connects particles across space",
                "depth": 3,
                "store_in_memory": True
            }
        )
        result = response.json()

        print(f"Root thought: {result['root_thought']['content']}")
        print(f"Coherence: {result['cascade_coherence']:.3f}")
        print(f"\nGenerated {len(result['generated_thoughts'])} thoughts:")

        for thought in result['generated_thoughts'][:5]:
            print(f"\n[Depth {thought['depth']}]")
            print(f"Content: {thought['content']}")
            print(f"Coherence: {thought['coherence_score']:.3f}")
            print(f"Entropy: {thought['entropy']:.3f}")

asyncio.run(basic_generation())
```

## Example 2: Quantum Superposition

```python
async def superposition_example():
    thoughts = [
        "Neural networks process information hierarchically",
        "Quantum computers leverage superposition for parallelism",
        "Holographic memory stores information distributively"
    ]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/newthought/superpose",
            json={"thought_texts": thoughts}
        )
        result = response.json()

        print("Superposed thought:")
        print(result['superposed_thought']['content'])
        print(f"\nCoherence: {result['superposed_thought']['coherence_score']:.3f}")

asyncio.run(superposition_example())
```

## Example 3: Memory Recall

```python
async def memory_recall():
    # First generate and store some thoughts
    await basic_generation()

    # Now recall similar thoughts
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/newthought/recall",
            json={
                "query_text": "quantum physics",
                "top_k": 5,
                "similarity_threshold": 0.4
            }
        )
        result = response.json()

        print(f"Found {len(result['similar_thoughts'])} similar thoughts:")
        for item in result['similar_thoughts']:
            thought = item['thought']
            similarity = item['similarity']
            print(f"\n[Similarity: {similarity:.3f}]")
            print(f"{thought['content']}")

asyncio.run(memory_recall())
```

## Example 4: Entanglement Measurement

```python
async def entanglement_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/newthought/entanglement",
            json={
                "thought_text_a": "Quantum superposition enables multiple states simultaneously",
                "thought_text_b": "Wave function collapse selects a single outcome"
            }
        )
        result = response.json()

        print(f"Quantum entanglement: {result['quantum_entanglement']:.3f}")
        print(f"Classical similarity: {result['classical_similarity']:.3f}")
        print(f"Interpretation: {result['interpretation']}")

asyncio.run(entanglement_example())
```

## Example 5: Service Statistics

```python
async def get_statistics():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/newthought/stats")
        stats = response.json()

        print("Service Statistics:")
        print(f"Total thoughts generated: {stats['service_stats']['total_thoughts_generated']}")
        print(f"Total cascades: {stats['service_stats']['total_cascades']}")
        print(f"Average coherence: {stats['service_stats']['avg_coherence']:.3f}")
        print(f"\nMemory Statistics:")
        print(f"Memory utilization: {stats['memory_stats']['memory_utilization']:.1%}")
        print(f"Average memory coherence: {stats['memory_stats']['avg_coherence']:.3f}")

asyncio.run(get_statistics())
```

## Example 6: Emergence Pattern Detection

```python
async def detect_emergence():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/newthought/generate",
            json={
                "seed_text": "Consciousness emerges from complex neural interactions",
                "depth": 5,  # Deeper for more emergence
                "store_in_memory": True
            }
        )
        result = response.json()

        print("Emergence Patterns Detected:")
        for pattern in result['emergence_patterns']:
            print(f"  - {pattern}")

        print(f"\nCascade coherence: {result['cascade_coherence']:.3f}")
        print(f"Depth reached: {result['depth_reached']}")

asyncio.run(detect_emergence())
```
