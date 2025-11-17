#!/usr/bin/env python3
"""
Integration Health Check
========================

Verifies all components of the Complete Unified LLM Platform are accessible
and properly connected. Tests imports, file existence, and service availability.

This script performs NO modifications - it only validates the integration.

Author: Claude Code
Date: 2025-11-08
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationHealthChecker:
    """Comprehensive health checker for all platform components"""

    def __init__(self):
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.kgirl_root = Path(__file__).parent

    def check_repository_structure(self) -> bool:
        """Verify all external repositories are cloned"""
        logger.info("=" * 70)
        logger.info("CHECKING REPOSITORY STRUCTURE")
        logger.info("=" * 70)

        required_repos = {
            "9xdSq-LIMPS-FemTO-R1C": "LIMPS GPU-accelerated optimization",
            "NuRea_sim": "Julia backend and entropy engine",
            "numbskull": "Fractal embeddings and neuro-symbolic engine"
        }

        all_present = True
        for repo_name, description in required_repos.items():
            repo_path = self.kgirl_root / repo_name
            exists = repo_path.exists() and repo_path.is_dir()

            status = "✅" if exists else "❌"
            logger.info(f"{status} {repo_name:30} - {description}")

            if not exists:
                self.errors.append(f"Missing repository: {repo_name}")
                all_present = False

        self.results['repos_structure'] = all_present
        return all_present

    def check_core_python_files(self) -> bool:
        """Verify core kgirl Python files exist"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING CORE PYTHON FILES")
        logger.info("=" * 70)

        required_files = [
            "complete_unified_platform.py",
            "unified_quantum_llm_system.py",
            "quantum_limps_integration.py",
            "quantum_holographic_knowledge_synthesis.py",
            "quantum_knowledge_database.py",
            "quantum_knowledge_processing.py",
            "main.py",
            "server.jl"
        ]

        all_present = True
        for filename in required_files:
            file_path = self.kgirl_root / filename
            exists = file_path.exists()

            status = "✅" if exists else "❌"
            logger.info(f"{status} {filename}")

            if not exists:
                self.errors.append(f"Missing core file: {filename}")
                all_present = False

        self.results['core_files'] = all_present
        return all_present

    def check_nurea_components(self) -> bool:
        """Verify NuRea_sim components"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING NUREA_SIM COMPONENTS")
        logger.info("=" * 70)

        nurea_path = self.kgirl_root / "NuRea_sim"

        if not nurea_path.exists():
            logger.error("❌ NuRea_sim directory not found")
            self.errors.append("NuRea_sim repository not cloned")
            self.results['nurea_components'] = False
            return False

        required_components = {
            "matrix_orchestrator.py": nurea_path / "matrix_orchestrator.py",
            "entropy_engine.py": nurea_path / "entropy engine" / "ent" / "entropy_engine.py",
        }

        all_present = True
        for name, path in required_components.items():
            exists = path.exists()
            status = "✅" if exists else "❌"
            logger.info(f"{status} {name:30} - {path.relative_to(self.kgirl_root)}")

            if not exists:
                self.errors.append(f"Missing NuRea component: {name}")
                all_present = False

        self.results['nurea_components'] = all_present
        return all_present

    def check_numbskull_components(self) -> bool:
        """Verify numbskull components"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING NUMBSKULL COMPONENTS")
        logger.info("=" * 70)

        numbskull_path = self.kgirl_root / "numbskull"

        if not numbskull_path.exists():
            logger.error("❌ numbskull directory not found")
            self.errors.append("numbskull repository not cloned")
            self.results['numbskull_components'] = False
            return False

        required_components = {
            "neuro_symbolic_engine.py": numbskull_path / "neuro_symbolic_engine.py",
            "fractal_cascade_embedder.py": numbskull_path / "advanced_embedding_pipeline" / "fractal_cascade_embedder.py",
            "holographic_similarity_engine.py": numbskull_path / "holographic_similarity_engine.py",
            "emergent_cognitive_network.py": numbskull_path / "emergent_cognitive_network.py",
        }

        all_present = True
        for name, path in required_components.items():
            exists = path.exists()
            status = "✅" if exists else "❌"
            rel_path = path.relative_to(self.kgirl_root) if path.exists() else path
            logger.info(f"{status} {name:35} - {rel_path}")

            if not exists:
                self.errors.append(f"Missing numbskull component: {name}")
                all_present = False

        self.results['numbskull_components'] = all_present
        return all_present

    def check_limps_components(self) -> bool:
        """Verify LIMPS framework components"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING LIMPS FRAMEWORK COMPONENTS")
        logger.info("=" * 70)

        limps_path = self.kgirl_root / "9xdSq-LIMPS-FemTO-R1C"

        if not limps_path.exists():
            logger.error("❌ 9xdSq-LIMPS-FemTO-R1C directory not found")
            self.errors.append("LIMPS repository not cloned")
            self.results['limps_components'] = False
            return False

        # Check for key directories
        key_dirs = ["limps_core", "matrix_ops", "entropy_analysis", "interfaces"]
        all_present = True

        for dir_name in key_dirs:
            dir_path = limps_path / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            status = "✅" if exists else "❌"
            logger.info(f"{status} {dir_name:25} - {dir_path.relative_to(self.kgirl_root)}")

            if not exists:
                self.warnings.append(f"Missing LIMPS directory: {dir_name}")
                all_present = False

        self.results['limps_components'] = all_present
        return all_present

    def test_imports(self) -> bool:
        """Test importing key modules"""
        logger.info("\n" + "=" * 70)
        logger.info("TESTING PYTHON IMPORTS")
        logger.info("=" * 70)

        # Add paths to sys.path
        sys.path.insert(0, str(self.kgirl_root))
        sys.path.insert(0, str(self.kgirl_root / "numbskull"))
        sys.path.insert(0, str(self.kgirl_root / "numbskull" / "advanced_embedding_pipeline"))
        sys.path.insert(0, str(self.kgirl_root / "NuRea_sim"))
        sys.path.insert(0, str(self.kgirl_root / "NuRea_sim" / "entropy engine"))

        test_imports = [
            ("quantum_holographic_knowledge_synthesis", "Core quantum system"),
            ("quantum_knowledge_database", "Quantum database"),
            ("quantum_limps_integration", "LIMPS integration"),
            ("unified_quantum_llm_system", "Unified system"),
        ]

        all_success = True
        for module_name, description in test_imports:
            try:
                importlib.import_module(module_name)
                logger.info(f"✅ {module_name:40} - {description}")
            except Exception as e:
                logger.error(f"❌ {module_name:40} - {description}")
                logger.error(f"   Error: {str(e)}")
                self.errors.append(f"Failed to import {module_name}: {e}")
                all_success = False

        # Test numbskull imports (may fail gracefully)
        numbskull_imports = [
            ("neuro_symbolic_engine", "Neuro-symbolic engine"),
            ("emergent_cognitive_network", "Emergent network"),
            ("holographic_similarity_engine", "Holographic engine"),
        ]

        for module_name, description in numbskull_imports:
            try:
                importlib.import_module(module_name)
                logger.info(f"✅ {module_name:40} - {description}")
            except Exception as e:
                logger.warning(f"⚠️  {module_name:40} - {description}")
                logger.warning(f"   Warning: {str(e)}")
                self.warnings.append(f"Optional import failed: {module_name}")

        self.results['imports'] = all_success
        return all_success

    def check_environment_variables(self) -> bool:
        """Check if required environment variables are set"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING ENVIRONMENT VARIABLES")
        logger.info("=" * 70)

        # Check if using local LLMs
        use_local_llm = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
        models_config = os.getenv("MODELS", "")
        using_ollama = "ollama:" in models_config or use_local_llm

        if using_ollama:
            logger.info("🚀 Using LOCAL LLM mode (Ollama) - API keys not required")
        else:
            logger.info("☁️  Using CLOUD API mode - API keys required")

        # Cloud API keys (optional when using Ollama)
        cloud_api_vars = {
            "OPENAI_API_KEY": "OpenAI API access (for cloud mode)",
            "ANTHROPIC_API_KEY": "Anthropic API access (for cloud mode)",
        }

        for var_name, description in cloud_api_vars.items():
            value = os.getenv(var_name)
            if value:
                masked = value[:8] + "..." if len(value) > 8 else "***"
                logger.info(f"✅ {var_name:25} - {description}")
                logger.info(f"   Value: {masked}")
            else:
                if using_ollama:
                    logger.info(f"➖ {var_name:25} - {description}")
                    logger.info(f"   Not needed (using Ollama)")
                else:
                    logger.warning(f"⚠️  {var_name:25} - {description}")
                    logger.warning(f"   Not set (required for cloud mode)")
                    self.warnings.append(f"Cloud API key not set: {var_name}")

        # Ollama configuration (when using local mode)
        if using_ollama:
            ollama_vars = {
                "OLLAMA_HOST": "Ollama service host",
                "OLLAMA_CHAT_MODEL": "Ollama chat model",
                "OLLAMA_EMBED_MODEL": "Ollama embedding model",
            }
            for var_name, description in ollama_vars.items():
                value = os.getenv(var_name)
                default_values = {
                    "OLLAMA_HOST": "http://localhost:11434",
                    "OLLAMA_CHAT_MODEL": "qwen2.5:3b",
                    "OLLAMA_EMBED_MODEL": "nomic-embed-text"
                }
                display_value = value if value else f"{default_values.get(var_name)} (default)"
                logger.info(f"✅ {var_name:25} - {description}")
                logger.info(f"   Value: {display_value}")

        # Database (optional)
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            logger.info(f"✅ DATABASE_URL           - PostgreSQL connection string")
        else:
            logger.info(f"➖ DATABASE_URL           - PostgreSQL connection (optional)")
            logger.info(f"   Not set (will use SQLite fallback)")

        # Other optional vars
        optional_vars = {
            "CTH_PATH": "Topological consciousness library path",
            "LIMPS_JULIA_URL": "LIMPS Julia service endpoint",
            "NUREA_JULIA_URL": "NuRea Julia backend endpoint",
            "CHAOS_RAG_URL": "ChaosRAG service endpoint"
        }

        for var_name, description in optional_vars.items():
            value = os.getenv(var_name)
            status = "✅" if value else "➖"
            logger.info(f"{status} {var_name:25} - {description}")

        # Always pass environment check (warnings only, no hard failures)
        self.results['environment'] = True
        return True

    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running (for local LLM mode)"""
        logger.info("\n" + "=" * 70)
        logger.info("CHECKING OLLAMA SERVICE (Local LLM)")
        logger.info("=" * 70)

        use_local_llm = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
        models_config = os.getenv("MODELS", "")
        using_ollama = "ollama:" in models_config or use_local_llm

        if not using_ollama:
            logger.info("➖ Ollama not configured (using cloud APIs)")
            self.results['ollama_service'] = True
            return True

        try:
            import requests
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            response = requests.get(f"{ollama_host}/api/version", timeout=5)
            if response.status_code == 200:
                version = response.json()
                logger.info(f"✅ Ollama service running at {ollama_host}")
                logger.info(f"   Version: {version}")
                self.results['ollama_service'] = True
                return True
            else:
                logger.error(f"❌ Ollama returned status {response.status_code}")
                self.errors.append(f"Ollama service check failed: status {response.status_code}")
                self.results['ollama_service'] = False
                return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama at {ollama_host}")
            logger.error(f"   Start Ollama with: ollama serve")
            self.errors.append("Ollama service not running")
            self.results['ollama_service'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Error checking Ollama: {e}")
            self.errors.append(f"Ollama check failed: {e}")
            self.results['ollama_service'] = False
            return False

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive health check report"""
        logger.info("\n" + "=" * 70)
        logger.info("INTEGRATION HEALTH CHECK REPORT")
        logger.info("=" * 70)

        total_checks = len(self.results)
        passed_checks = sum(1 for v in self.results.values() if v)

        logger.info(f"\nOverall Status: {passed_checks}/{total_checks} checks passed")
        logger.info("\nDetailed Results:")

        for check_name, passed in self.results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {status:10} - {check_name}")

        if self.errors:
            logger.error(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        all_critical_passed = (
            self.results.get('repos_structure', False) and
            self.results.get('core_files', False) and
            self.results.get('nurea_components', False) and
            self.results.get('numbskull_components', False)
        )

        if all_critical_passed:
            logger.info("\n🎉 SUCCESS: All critical components are present!")
            logger.info("   The Complete Unified Platform is ready for integration.")
        else:
            logger.error("\n❌ FAILURE: Critical components are missing!")
            logger.error("   Review errors above and ensure all repositories are cloned.")

        return {
            'passed': all_critical_passed,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings
        }

    def run_full_check(self) -> Dict[str, Any]:
        """Run all health checks"""
        self.check_repository_structure()
        self.check_core_python_files()
        self.check_nurea_components()
        self.check_numbskull_components()
        self.check_limps_components()
        self.test_imports()
        self.check_environment_variables()
        self.check_ollama_service()

        return self.generate_report()


def main():
    """Main execution"""
    print("=" * 70)
    print("COMPLETE UNIFIED LLM PLATFORM - INTEGRATION HEALTH CHECK")
    print("=" * 70)
    print("This script validates all components are accessible and connected.")
    print()

    checker = IntegrationHealthChecker()
    report = checker.run_full_check()

    # Exit code
    sys.exit(0 if report['passed'] else 1)


if __name__ == "__main__":
    main()
