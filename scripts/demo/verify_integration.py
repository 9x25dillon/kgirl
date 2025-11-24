#!/usr/bin/env python3
"""
Integration Verification Script
================================

Quick verification that all components are properly set up
for the LFM2 + Numbskull + Dual LLM integration.

Usage:
    python verify_integration.py

Author: Assistant
License: MIT
"""

import sys
import json
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False


def check_module_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description} IMPORT FAILED: {module_name}")
        print(f"   Error: {e}")
        return False


def check_numbskull_components():
    """Check numbskull components availability"""
    sys.path.insert(0, "/home/kill/numbskull")
    
    components = [
        ("advanced_embedding_pipeline", "Numbskull Base Package"),
        ("advanced_embedding_pipeline.hybrid_pipeline", "Hybrid Pipeline"),
        ("advanced_embedding_pipeline.semantic_embedder", "Semantic Embedder"),
        ("advanced_embedding_pipeline.mathematical_embedder", "Mathematical Embedder"),
        ("advanced_embedding_pipeline.fractal_cascade_embedder", "Fractal Embedder"),
    ]
    
    results = []
    for module, desc in components:
        results.append(check_module_import(module, desc))
    
    return all(results)


def check_service_connectivity():
    """Check if services are reachable"""
    try:
        import requests
    except ImportError:
        print("⚠️  requests module not available for service checks")
        return True
    
    services = [
        ("http://127.0.0.1:8080", "LFM2-8B-A1B (Local LLM)", "/v1/models"),
        ("http://127.0.0.1:8001", "Eopiez (Semantic)", "/health"),
        ("http://127.0.0.1:8000", "LIMPS (Mathematical)", "/health"),
    ]
    
    print("\n" + "=" * 60)
    print("SERVICE CONNECTIVITY CHECK")
    print("=" * 60)
    
    for base_url, name, endpoint in services:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=2)
            if response.status_code < 500:
                print(f"✅ {name}: {base_url} (reachable)")
            else:
                print(f"⚠️  {name}: {base_url} (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠️  {name}: {base_url} (not reachable - {type(e).__name__})")
            print(f"   Note: This is optional. System will use fallback.")
    
    return True


def verify_config():
    """Verify configuration file"""
    config_path = Path("/home/kill/LiMp/config_lfm2.json")
    
    if not config_path.exists():
        print("⚠️  config_lfm2.json not found (will use defaults)")
        return True
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        print(f"✅ Configuration file valid: {config_path}")
        
        # Check key sections
        if "local_llm" in config:
            llm = config["local_llm"]
            print(f"   Local LLM: {llm.get('model', 'N/A')} @ {llm.get('base_url', 'N/A')}")
        
        if "orchestrator_settings" in config:
            settings = config["orchestrator_settings"]
            print(f"   Numbskull enabled: {settings.get('use_numbskull', False)}")
            print(f"   Fusion method: {settings.get('fusion_method', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        return False


def main():
    """Main verification routine"""
    print("=" * 60)
    print("LFM2 + NUMBSKULL + DUAL LLM INTEGRATION VERIFICATION")
    print("=" * 60)
    print()
    
    results = []
    
    # Check core files
    print("CORE FILES")
    print("-" * 60)
    results.append(check_file_exists(
        "/home/kill/LiMp/numbskull_dual_orchestrator.py",
        "Numbskull Orchestrator"
    ))
    results.append(check_file_exists(
        "/home/kill/LiMp/dual_llm_orchestrator.py",
        "Base Dual Orchestrator"
    ))
    results.append(check_file_exists(
        "/home/kill/LiMp/run_integrated_workflow.py",
        "Workflow Runner"
    ))
    results.append(check_file_exists(
        "/home/kill/LiMp/config_lfm2.json",
        "Configuration File"
    ))
    results.append(check_file_exists(
        "/home/kill/LiMp/README_INTEGRATION.md",
        "Integration Documentation"
    ))
    
    print()
    
    # Check numbskull availability
    print("NUMBSKULL COMPONENTS")
    print("-" * 60)
    numbskull_ok = check_numbskull_components()
    results.append(numbskull_ok)
    
    print()
    
    # Check configuration
    print("CONFIGURATION")
    print("-" * 60)
    config_ok = verify_config()
    results.append(config_ok)
    
    print()
    
    # Check services (optional)
    check_service_connectivity()
    
    print()
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ ALL CRITICAL COMPONENTS VERIFIED")
        print()
        print("Next steps:")
        print("1. Start LFM2-8B-A1B server on http://127.0.0.1:8080")
        print("2. Run demo: python run_integrated_workflow.py --demo")
        print("3. Or interactive: python run_integrated_workflow.py --interactive")
        print()
        print("Optional services (use fallbacks if unavailable):")
        print("- Eopiez (semantic): http://127.0.0.1:8001")
        print("- LIMPS (mathematical): http://127.0.0.1:8000")
        return 0
    else:
        print("❌ SOME COMPONENTS MISSING OR FAILED")
        print()
        print("Please check the errors above and:")
        print("1. Ensure numbskull is installed: pip install -e /home/kill/numbskull")
        print("2. Verify all files are present in /home/kill/LiMp")
        print("3. Check requirements.txt and install dependencies")
        return 1


if __name__ == "__main__":
    sys.exit(main())

