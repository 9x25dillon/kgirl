import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Any

# -------------------------------
# KFPLayer: stable, differentiable
# -------------------------------
class KFPLayer(nn.Module):
    """
    Kinetic Force Principle Layer
    Tracks per-feature fluctuation (variance) with momentum and applies a
    projection-based damping proportional to inverse variance.
    """
    def __init__(self, dim: int, stability_weight: float = 0.1, momentum: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.stability_weight = stability_weight
        self.momentum = momentum
        self.eps = eps
        self.force_projection = nn.Linear(dim, dim, bias=False)
        # Running variance buffer (no grad)
        self.register_buffer("fluctuation_history", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., dim)
        Returns (stabilized_x, fluctuation_history)
        """
        # Compute current variance across all non-feature axes
        flat = x.reshape(-1, x.shape[-1])  # (N, D)
        current_var = flat.var(dim=0, unbiased=False)  # (D,)

        # Momentum update (no grad)
        with torch.no_grad():
            self.fluctuation_history.mul_(self.momentum).add_((1.0 - self.momentum) * current_var)

        # Inverse-variance gain (normalized for stability)
        gains = 1.0 / (self.fluctuation_history + self.eps)        # (D,)
        gains = gains / gains.mean().clamp_min(self.eps)           # (D,)

        # Project then damp features with higher fluctuation more strongly
        proj = self.force_projection(x)                            # (..., D)
        stability_term = -self.stability_weight * proj * gains     # broadcast (..., D) * (D,)

        return x + stability_term, self.fluctuation_history.clone()

# -------------------------------------
# TAULSControlUnit without broken Seq’s
# -------------------------------------
class TAULSControlUnit(nn.Module):
    """
    Two-level Trans-Algorithmic Universal Learning System
    Higher level: Learning (meta-control)
    Lower level: Automatic control
    """
    def __init__(self, input_dim: int, hidden_dim: int, control_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        # Meta path
        self.meta_in = nn.Sequential(
            nn.Linear(input_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.meta_kfp = KFPLayer(hidden_dim)
        self.meta_out = nn.Linear(hidden_dim, control_dim)

        # Auto path
        h2 = max(2, hidden_dim // 2)
        self.auto_in = nn.Sequential(
            nn.Linear(input_dim, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
        )
        self.auto_kfp = KFPLayer(h2)
        self.auto_out = nn.Linear(h2, control_dim)

        # Learnable mixer
        self.control_mixer = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, prev_control: Optional[torch.Tensor] = None) -> Dict:
        """
        x: (B, S, input_dim)
        prev_control: (B, S, control_dim) or None
        """
        B, S, Din = x.shape
        if prev_control is None:
            prev_control = torch.zeros(B, S, self.control_dim, device=x.device, dtype=x.dtype)

        # Meta path
        meta_inp = torch.cat([x, prev_control], dim=-1).reshape(-1, Din + self.control_dim)
        meta_h = self.meta_in(meta_inp)               # (B*S, H)
        meta_h, meta_stab = self.meta_kfp(meta_h)     # (B*S, H), (D,)
        meta_ctl = self.meta_out(meta_h).reshape(B, S, self.control_dim)

        # Auto path
        auto_inp = x.reshape(-1, Din)                 # (B*S, Din)
        auto_h = self.auto_in(auto_inp)               # (B*S, H2)
        auto_h, auto_stab = self.auto_kfp(auto_h)     # (B*S, H2), (D2,)
        auto_ctl = self.auto_out(auto_h).reshape(B, S, self.control_dim)

        # Mix
        alpha = torch.sigmoid(self.control_mixer)
        integrated = alpha * meta_ctl + (1 - alpha) * auto_ctl

        return {
            "control_output": integrated,
            "meta_stability": meta_stab,
            "auto_stability": auto_stab,
            "control_mixing": alpha
        }

# ----------------------------------------
# EntropyRegulationModule shape-safe update
# ----------------------------------------
class EntropyRegulationModule(nn.Module):
    """
    Entropy regulation based on environmental stress.
    Modulates per-feature scale to maintain active stability.
    """
    def __init__(self, dim: int, max_entropy_target: float = 0.8):
        super().__init__()
        self.dim = dim
        self.max_entropy_target = max_entropy_target

        self.entropy_estimator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        self.intensity_controller = nn.Linear(1, dim)

    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar estimate (mean across batch/seq).
        """
        flat = x.reshape(-1, x.shape[-1])            # (N, D)
        est = self.entropy_estimator(flat).squeeze(-1)  # (N,)
        return est.mean()

    def forward(self, x: torch.Tensor, environmental_stress: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        x: (B, S, D)
        environmental_stress: any shape → reduced to scalar mean
        """
        current_entropy = self.compute_entropy(x)     # scalar
        stress_factor = environmental_stress.mean()   # scalar

        # Scalar target intensity in [0,1]
        entropy_error = current_entropy - self.max_entropy_target
        target_intensity = torch.sigmoid(entropy_error + stress_factor).view(1, 1)  # (1,1)

        # Map to per-feature scale (1, D) → broadcast to (1,1,D)
        scales = self.intensity_controller(target_intensity)         # (1, D)
        scales = torch.sigmoid(scales).view(1, 1, self.dim)          # keep it bounded (0,1)

        modulated = x * scales
        return modulated, {
            "current_entropy": current_entropy.detach(),
            "target_intensity": target_intensity.detach(),
            "entropy_error": entropy_error.detach()
        }

# ------------------------------------------
# TAULS Transformer block with fixed wiring
# ------------------------------------------
class TAULSTransformerBlock(nn.Module):
    """
    Transformer block enhanced with TA ULS control structure.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.control_unit = TAULSControlUnit(d_model, d_ff, d_model)
        self.entropy_regulator = EntropyRegulationModule(d_model)
        self.stability_layer = KFPLayer(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict:
        """
        x: (B, S, D)
        mask: attention mask (see torch.nn.MultiheadAttention docs)
        """
        attn_output, attn_weights = self.self_attention(x, x, x, attn_mask=mask)  # (B,S,D), (B,heads,S,S)
        x_res = self.norm1(x + self.dropout(attn_output))

        # Environmental stress from attention statistics
        env_stress = attn_weights.var(dim=-1, unbiased=False).mean(dim=-1, keepdim=True)  # (B, H, 1)

        # Entropy regulation
        regulated_x, entropy_info = self.entropy_regulator(x_res, env_stress)

        # TA ULS control
        control_results = self.control_unit(regulated_x)
        controlled_x = control_results["control_output"]

        # KFP stability
        stable_x, fluct_int = self.stability_layer(controlled_x)

        # Residual + norm
        out = self.norm2(x_res + self.dropout(stable_x))

        return {
            "output": out,
            "attention_weights": attn_weights,
            "control_info": control_results,
            "entropy_info": entropy_info,
            "stability_info": fluct_int
        }

# --------------------------------
# Full TAULS language model (LM)
# --------------------------------
class TAULSLanguageModel(nn.Module):
    """
    Complete language model implementing TA ULS architecture.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TAULSTransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.global_stability_tracker = KFPLayer(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict:
        """
        input_ids: (B,S)
        """
        B, S = input_ids.shape
        device = input_ids.device

        tok = self.token_embedding(input_ids)                                 # (B,S,D)
        pos = self.position_embedding(torch.arange(S, device=device)).unsqueeze(0)  # (1,S,D)
        x = tok + pos

        layer_outputs: List[torch.Tensor] = []
        stability_metrics: List[Dict] = []

        for i, block in enumerate(self.blocks):
            block_res = block(x, attention_mask)
            x = block_res["output"]
            layer_outputs.append(x)
            stability_metrics.append({
                "layer": i,
                "control_info": block_res["control_info"],
                "entropy_info": block_res["entropy_info"],
                "stability_info": block_res["stability_info"],
            })

        stable_x, global_stability = self.global_stability_tracker(x)
        logits = self.output_projection(stable_x)

        return {
            "logits": logits,
            "hidden_states": layer_outputs,
            "stability_metrics": stability_metrics,
            "global_stability": global_stability
        }

# -------------------------------------------------
# Polynomial KFP helper (kept, but shape-corrected)
# -------------------------------------------------
def create_kfp_polynomial_basis(degree: int, dim: int) -> torch.Tensor:
    """
    Create polynomial basis coefficients for a simple potential f(x).
    coefficients[d] is a (dim, dim) matrix for the degree-d term.
    Quadratic terms are made negative-definite-ish by clamping diagonal.
    """
    coeffs = torch.randn(degree + 1, dim, dim) * 0.1
    # Make quadratic term diagonally negative
    if degree >= 2:
        with torch.no_grad():
            q = coeffs[2]
            q.copy_(q - 2.0 * torch.diag(torch.abs(torch.diag(q)) + 1e-3))
    return coeffs

def kfp_polynomial_update(x: torch.Tensor, coefficients: torch.Tensor, learning_rate: float = 0.01) -> torch.Tensor:
    """
    Polynomial-based KFP update rule:
      dx/dt = -∇f(x)
    Using a matrix polynomial f(x) ~ Σ_d x^{d-1} * C[d]  (toy illustrative).
    """
    *lead, D = x.shape
    flat = x.reshape(-1, D)  # (N, D)
    degree = coefficients.shape[0] - 1
    grad = torch.zeros_like(flat)
    for d in range(1, degree + 1):
        # grad term ≈ d * (x^(d-1) @ C_d)
        xd1 = flat.pow(d - 1)                               # (N, D)
        Cd = coefficients[d]                                # (D, D)
        grad += d * (xd1 @ Cd)                              # (N, D)
    flat = flat - learning_rate * grad
    return flat.reshape(*lead, D)


# --------------------------
# TAULS Evaluator wrapper
# --------------------------
class TAULSEvaluator:
    """Thin evaluator wrapper exposing a .score(text) method.

    This implementation uses a lightweight character-level entropy/stability
    heuristic so it runs quickly without heavy tokenizers. It mirrors the
    fallback used in `llm_eval.py` but centralizes the interface for callers.
    """
    def __init__(self):
        pass

    def score(self, text: str) -> Dict[str, float]:
        # basic char-level entropy
        cnt = {}
        for ch in text:
            cnt[ch] = cnt.get(ch, 0) + 1
        total = len(text) or 1
        entropy = -sum((v/total) * math.log((v/total)+1e-12, 2) for v in cnt.values())
        unique = max(1, len(cnt))
        max_ent = math.log(unique, 2) if unique > 1 else 1.0
        entropy_norm = entropy / max_ent if max_ent > 0 else 0.0

        punct = sum(1 for c in text if c in '.!?')
        stability = 1.0 - min(1.0, punct / (total / 10 + 1e-12))

        return {'entropy': float(entropy_norm), 'stability': float(stability), 'length': int(total)}


# --------------------------
# TAULS Runner (model-based scorer)
# --------------------------
class TAULSRunner:
    """Lightweight runner that instantiates a small TAULSLanguageModel and
    provides a score_text(text) method returning model-based metrics.

    This is intentionally small (configurable) so it can run on CPU for evaluations.
    """
    def __init__(self, vocab_size: int = 1024, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 128, device: Optional[str] = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # instantiate a small model
        self.model = TAULSLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(self.device)
        self.model.eval()

    def _tokenize(self, text: str):
        # Simple byte-level tokenizer mapping ord(char) % vocab
        ids = [ord(c) % self.vocab_size for c in text]
        # truncate/pad
        if len(ids) > self.max_seq_len:
            ids = ids[-self.max_seq_len:]
        else:
            ids = [0] * (self.max_seq_len - len(ids)) + ids
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def score_text(self, text: str) -> Dict[str, Any]:
        with torch.no_grad():
            input_ids = self._tokenize(text)
            out = self.model(input_ids)
            logits = out['logits']  # (B, S, V)
            # compute token-wise entropy (base e -> convert to bits later)
            probs = F.softmax(logits.float(), dim=-1)
            logp = torch.where(probs > 0, torch.log(probs), torch.tensor(0.0, device=probs.device))
            token_entropy = -torch.sum(probs * logp, dim=-1)  # (B, S)
            # average non-padding tokens (we used zero padding tokens)
            avg_entropy = float(token_entropy.mean().item())

            # global stability metric from model (KFPLayer output)
            global_stab = out.get('global_stability')
            if global_stab is not None:
                stab_val = float(global_stab.abs().mean().item())
            else:
                stab_val = 0.0

            # approximate perplexity from entropy
            ppl = float(torch.exp(torch.tensor(avg_entropy)).item())

            return {
                'model_entropy_nats': avg_entropy,
                'perplexity': ppl,
                'global_stability': stab_val,
                'length': len(text)
            }

# ----------------
# Quick sanity run
# ----------------
if __name__ == "__main__":
    torch.set_grad_enabled(True)
    vocab_size = 50000
    d_model = 256
    n_heads = 8
    n_layers = 3
    max_seq_len = 512
    model = TAULSLanguageModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    out = model(input_ids)
    print("Logits:", tuple(out["logits"].shape))
    print("Layers:", len(out["stability_metrics"]))
    print("Global stability:", tuple(out["global_stability"].shape))
    # Poly KFP sanity
    coeffs = create_kfp_polynomial_basis(degree=3, dim=d_model)
    dummy = torch.randn(batch_size, seq_len, d_model)
    upd = kfp_polynomial_update(dummy, coeffs, learning_rate=0.01)
    print("Poly update shape:", tuple(upd.shape))
