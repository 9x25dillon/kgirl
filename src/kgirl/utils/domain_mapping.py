# domain_mapping.py
from dataclasses import dataclass
from typing import Dict, Callable

@dataclass
class DomainMapping:
    entity_mappings: Dict[str, str] = None
    value_transformations: Dict[str, Callable] = None
    inverse_transformations: Dict[str, Callable] = None
    aggregation_functions: Dict[str, Callable] = None
    distribution_functions: Dict[str, Callable] = None

    def __post_init__(self):
        if self.entity_mappings is None:
            self.entity_mappings = {}
        if self.value_transformations is None:
            self.value_transformations = {}
        # ... etc

# === ACTUAL MAPPING ===
domain_mapping = DomainMapping()

# Forward: Thought → EEG
domain_mapping.distribution_functions["coherence_to_bands"] = lambda c, m: {
    'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0
}  # placeholder — real one below