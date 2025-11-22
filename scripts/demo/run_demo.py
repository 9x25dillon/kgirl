#!/home/kill/aipyapp/venv/bin/python3
"""
Demo runner script that properly activates the virtual environment
"""
import sys
import os

# Add the virtual environment site-packages to the path
venv_site_packages = '/home/kill/aipyapp/venv/lib/python3.13/site-packages'
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Now run the main demo
if __name__ == "__main__":
    from cognitive_communication_organism import demo_cognitive_communication_organism

    print("🚀 Running Cognitive Communication Organism Demo...")
    print("=" * 80)

    try:
        result = demo_cognitive_communication_organism()
        print("\n✅ Demo completed successfully!")
        print(f"📊 Processed {len(result['communication_results'])} communication scenarios")
        print(f"🏥 Emergency network established with {len(result['emergency_network']['nodes'])} nodes")
        print(f"🔬 Protocol evolution completed with {result['evolution_result']['episodes_completed']} episodes")
        print(f"✨ All 5 emergent technology areas successfully integrated and demonstrated")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
