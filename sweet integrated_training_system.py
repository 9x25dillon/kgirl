#!/usr/bin/env python3
"""
Enhanced TA ULS Training System with Julia Integration
Implements stability-aware training with polynomial optimization backend
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import subprocess
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import requests
from pathlib import Path

# Import Julia client from the provided code
class JuliaClient:
    """Python client for Julia mathematical operations"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
        
    def _make_request(self, function_name: str, args: List[Any]) -> Dict[str, Any]:
        try:
            payload = {"function": function_name, "args": args}
            response = self.session.post(
                self.server_url, json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response.json() if response.status_code == 200 else {"error": f"Server error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {e}"}
    
    def optimize_matrix(self, matrix: np.ndarray, method: str = "sparsity") -> Dict[str, Any]:
        return self._make_request("optimize_matrix", [matrix.tolist(), method])
    
    def create_polynomials(self, data: np.ndarray, variables: List[str]) -> Dict[str, Any]:
        return self._make_request("create_polynomials", [data.tolist(), variables])
    
    def analyze_polynomials(self, polynomials: Dict[str, Any]) -> Dict[str, Any]:
        return self._make_request("analyze_polynomials", [polynomials])

# Import the TA ULS model components from previous artifact
class KFPLayer(nn.Module):
    def __init__(self, dim: int, stability_weight: float = 0.1):
        super().__init__()
        self.dim = dim
        self.stability_weight = stability_weight
        self.fluctuation_history = nn.Parameter(torch.zeros(dim), requires_grad=False)
        self.momentum = 0.9
        self.force_projection = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_fluctuation = torch.var(x, dim=0, keepdim=False)
        self.fluctuation_history.data = (
            self.momentum * self.fluctuation_history.data + 
            (1 - self.momentum) * current_fluctuation.detach()
        )
        kinetic_force = self.force_projection(x)
        stability_term = -self.stability_weight * kinetic_force
        return x + stability_term, self.fluctuation_history

class TAULSControlUnit(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, control_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim
        
        self.meta_controller = nn.Sequential(
            nn.Linear(input_dim + control_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            KFPLayer(hidden_dim),
            nn.Linear(hidden_dim, control_dim)
        )
        
        self.controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            KFPLayer(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, control_dim)
        )
        
        self.control_mixer = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor, prev_control: Optional[torch.Tensor] = None) -> Dict:
        batch_size, seq_len = x.shape[:2]
        
        if prev_control is None:
            prev_control = torch.zeros(batch_size, seq_len, self.control_dim, device=x.device)
        
        meta_input = torch.cat([x, prev_control], dim=-1)
        meta_control, meta_stability = self.meta_controller[:-1](meta_input.reshape(-1, meta_input.shape[-1]))
        meta_control = self.meta_controller[-1](meta_control[0]).reshape(batch_size, seq_len, -1)
        
        auto_control, auto_stability = self.controller[:-1](x.reshape(-1, x.shape[-1]))
        auto_control = self.controller[-1](auto_control[0]).reshape(batch_size, seq_len, -1)
        
        alpha = torch.sigmoid(self.control_mixer)
        integrated_control = alpha * meta_control + (1 - alpha) * auto_control
        
        return {
            'control_output': integrated_control,
            'meta_stability': meta_stability,
            'auto_stability': auto_stability,
            'control_mixing': alpha
        }

@dataclass
class TAULSTrainingConfig:
    """Configuration for TA ULS training"""
    vocab_size: int = 50000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 2048
    batch_size: int = 8
    learning_rate: float = 1e-4
    stability_weight: float = 0.1
    entropy_weight: float = 0.05
    julia_server_port: int = 8000
    use_julia_optimization: bool = True
    optimization_frequency: int = 100  # Optimize every N steps
    stability_threshold: float = 0.8
    max_entropy_target: float = 0.7

class JuliaServerManager:
    """Manages Julia server lifecycle"""
    
    def __init__(self, port: int = 8000, julia_script_path: str = "julia_integration.jl"):
        self.port = port
        self.julia_script_path = julia_script_path
        self.process = None
        self.client = None
        
    def start_server(self):
        """Start Julia HTTP server"""
        try:
            # Start Julia server in background
            cmd = [
                "julia", "-e", 
                f"""
                include("{self.julia_script_path}")
                start_http_server({self.port})
                """
            ]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(3)
            
            # Create client
            self.client = JuliaClient(f"http://localhost:{self.port}")
            
            # Test connection
            test_matrix = np.random.rand(3, 3)
            result = self.client.optimize_matrix(test_matrix)
            
            if "error" not in result:
                logging.info(f"Julia server started successfully on port {self.port}")
                return True
            else:
                logging.error(f"Julia server connection test failed: {result}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to start Julia server: {e}")
            return False
    
    def stop_server(self):
        """Stop Julia server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logging.info("Julia server stopped")

class StabilityAwareLoss(nn.Module):
    """Custom loss function incorporating KFP stability principles"""
    
    def __init__(self, entropy_weight: float = 0.05, stability_weight: float = 0.1):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.stability_weight = stability_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def compute_entropy_loss(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """Compute entropy regularization loss"""
        total_entropy = 0.0
        for hidden in hidden_states:
            # Approximate entropy using variance
            entropy = torch.var(hidden, dim=-1).mean()
            total_entropy += entropy
        return total_entropy / len(hidden_states)
    
    def compute_stability_loss(self, stability_metrics: List[Dict]) -> torch.Tensor:
        """Compute stability loss based on fluctuation intensity"""
        total_stability_loss = 0.0
        count = 0
        
        for metrics in stability_metrics:
            if 'stability_info' in metrics:
                stability_info = metrics['stability_info']
                # Penalize high fluctuation intensity
                stability_loss = torch.mean(stability_info)
                total_stability_loss += stability_loss
                count += 1
        
        return total_stability_loss / max(count, 1)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                hidden_states: List[torch.Tensor], 
                stability_metrics: List[Dict]) -> Dict[str, torch.Tensor]:
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        # Entropy regularization (encourage controlled entropy)
        entropy_loss = self.compute_entropy_loss(hidden_states)
        
        # Stability loss (penalize high fluctuation intensity)
        stability_loss = self.compute_stability_loss(stability_metrics)
        
        # Combined loss
        total_loss = (ce_loss + 
                     self.entropy_weight * entropy_loss + 
                     self.stability_weight * stability_loss)
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'entropy_loss': entropy_loss,
            'stability_loss': stability_loss
        }

class TAULSOptimizer:
    """Julia-enhanced optimizer for TA ULS model"""
    
    def __init__(self, model: nn.Module, config: TAULSTrainingConfig, julia_client: JuliaClient):
        self.model = model
        self.config = config
        self.julia_client = julia_client
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.step_count = 0
        
    def optimize_parameters_with_julia(self):
        """Use Julia backend to optimize model parameters"""
        try:
            optimized_params = {}
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Convert parameter to numpy
                    param_np = param.data.cpu().numpy()
                    
                    # Skip if parameter is too small or 1D
                    if param_np.size < 4 or len(param_np.shape) < 2:
                        continue
                    
                    # Reshape to 2D if needed
                    original_shape = param_np.shape
                    if len(param_np.shape) > 2:
                        param_2d = param_np.reshape(param_np.shape[0], -1)
                    else:
                        param_2d = param_np
                    
                    # Optimize using Julia
                    result = self.julia_client.optimize_matrix(param_2d, method="sparsity")
                    
                    if "error" not in result and "optimized_matrix" in result:
                        # Reshape back to original shape
                        optimized_param = np.array(result["optimized_matrix"])
                        if len(original_shape) > 2:
                            optimized_param = optimized_param.reshape(original_shape)
                        
                        # Apply optimization (weighted combination)
                        alpha = 0.1  # Conservative mixing
                        param.data = torch.tensor(
                            (1 - alpha) * param_np + alpha * optimized_param,
                            dtype=param.dtype, device=param.device
                        )
                        
                        optimized_params[name] = {
                            'compression_ratio': result.get('compression_ratio', 0.0),
                            'optimization_method': 'sparsity'
                        }
            
            logging.info(f"Julia optimization applied to {len(optimized_params)} parameters")
            return optimized_params
            
        except Exception as e:
            logging.warning(f"Julia optimization failed: {e}")
            return {}
    
    def step(self, loss: torch.Tensor) -> Dict[str, Any]:
        """Perform optimization step with optional Julia enhancement"""
        self.optimizer.zero_grad()
        loss.backward()
        
        # Standard gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Apply Julia optimization periodically
        optimization_info = {}
        if (self.config.use_julia_optimization and 
            self.step_count % self.config.optimization_frequency == 0):
            optimization_info = self.optimize_parameters_with_julia()
        
        self.optimizer.step()
        self.step_count += 1
        
        return {
            'step': self.step_count,
            'julia_optimization': optimization_info,
            'gradient_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
        }

class TAULSTrainer:
    """Main trainer class for TA ULS model"""
    
    def __init__(self, config: TAULSTrainingConfig):
        self.config = config
        self.julia_manager = JuliaServerManager(port=config.julia_server_port)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model (simplified version for demonstration)
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = StabilityAwareLoss(
            entropy_weight=config.entropy_weight,
            stability_weight=config.stability_weight
        )
        
        # Will be initialized after Julia server starts
        self.optimizer = None
        
    def _create_model(self) -> nn.Module:
        """Create simplified TA ULS model for demonstration"""
        class SimpleTA_ULS(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embedding = nn.Embedding(config.vocab_size, config.d_model)
                self.control_unit = TAULSControlUnit(config.d_model, config.d_model, config.d_model)
                self.output = nn.Linear(config.d_model, config.vocab_size)
                self.kfp_layer = KFPLayer(config.d_model)
                
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                control_result = self.control_unit(x)
                controlled_x = control_result['control_output']
                stable_x, stability_info = self.kfp_layer(controlled_x)
                logits = self.output(stable_x)
                
                return {
                    'logits': logits,
                    'hidden_states': [x, controlled_x, stable_x],
                    'stability_metrics': [{'stability_info': stability_info}],
                    'control_info': control_result
                }
        
        return SimpleTA_ULS(self.config)
    
    def start_training(self):
        """Initialize training environment"""
        # Start Julia server
        if self.config.use_julia_optimization:
            if not self.julia_manager.start_server():
                logging.warning("Julia server failed to start, disabling Julia optimization")
                self.config.use_julia_optimization = False
        
        # Initialize optimizer
        julia_client = self.julia_manager.client if self.config.use_julia_optimization else None
        self.optimizer = TAULSOptimizer(self.model, self.config, julia_client)
        
        logging.info("Training environment initialized")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # Forward pass
        model_output = self.model(input_ids)
        
        # Compute loss
        loss_info = self.loss_fn(
            model_output['logits'],
            targets,
            model_output['hidden_states'],
            model_output['stability_metrics']
        )
        
        # Optimization step
        opt_info = self.optimizer.step(loss_info['total_loss'])
        
        return {
            'loss': loss_info,
            'optimization': opt_info,
            'model_output': {k: v for k, v in model_output.items() if k != 'logits'}
        }
    
    def evaluate_stability(self) -> Dict[str, float]:
        """Evaluate model stability metrics"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate random input for stability testing
            test_input = torch.randint(0, self.config.vocab_size, 
                                     (self.config.batch_size, 64), 
                                     device=self.device)
            
            outputs = []
            for _ in range(10):  # Multiple forward passes
                output = self.model(test_input)
                outputs.append(output)
            
            # Compute stability metrics
            logit_variance = torch.var(torch.stack([o['logits'] for o in outputs]), dim=0).mean()
            
            stability_scores = []
            for output in outputs:
                for metric in output['stability_metrics']:
                    if 'stability_info' in metric:
                        stability_scores.append(metric['stability_info'].mean().item())
            
            return {
                'logit_stability': logit_variance.item(),
                'mean_stability_score': np.mean(stability_scores) if stability_scores else 0.0,
                'stability_variance': np.var(stability_scores) if stability_scores else 0.0
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'julia_manager'):
            self.julia_manager.stop_server()

def create_dummy_dataset(config: TAULSTrainingConfig, num_samples: int = 1000):
    """Create dummy dataset for demonstration"""
    data = []
    for _ in range(num_samples):
        seq_len = np.random.randint(10, min(config.max_seq_len, 100))
        input_ids = torch.randint(0, config.vocab_size, (seq_len,))
        targets = torch.randint(0, config.vocab_size, (seq_len,))
        data.append({'input_ids': input_ids, 'targets': targets})
    return data

def main():
    """Main training demonstration"""
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = TAULSTrainingConfig(
        vocab_size=1000,  # Smaller for demo
        d_model=128,      # Smaller for demo
        batch_size=4,
        use_julia_optimization=True,
        optimization_frequency=10
    )
    
    # Create trainer
    trainer = TAULSTrainer(config)
    
    try:
        # Start training environment
        trainer.start_training()
        
        # Create dummy dataset
        dataset = create_dummy_dataset(config, num_samples=100)
        
        # Training loop
        for epoch in range(3):
            logging.info(f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(dataset[:20]):  # Limited steps for demo
                # Pad batch for consistent tensor shapes
                batch_data = {
                    'input_ids': batch['input_ids'][:50].unsqueeze(0),  # Add batch dim
                    'targets': batch['targets'][:50].unsqueeze(0)
                }
                
                # Training step
                step_result = trainer.train_step(batch_data)
                
                if step % 5 == 0:
                    logging.info(f"Step {step}: Loss = {step_result['loss']['total_loss'].item():.4f}")
            
            # Evaluate stability
            stability = trainer.evaluate_stability()
            logging.info(f"Epoch {epoch + 1} Stability: {stability}")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()