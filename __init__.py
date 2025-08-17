"""
Intelligent Self-Optimizing Nodes for ComfyUI
Transforms workflows into self-improving machines that learn with each run
"""

import sys
import traceback

print("[INTELLIGENT_NODES] Starting to load Intelligent Nodes...")

# Import intelligent self-optimizing nodes
try:
    print("[INTELLIGENT_NODES] Attempting to import from intelligent_nodes.py...")
    from .intelligent_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    INTELLIGENT_AVAILABLE = True
    print(f"[INTELLIGENT_NODES] Successfully imported {len(NODE_CLASS_MAPPINGS)} nodes")
    print(f"[INTELLIGENT_NODES] Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
except ImportError as e:
    INTELLIGENT_AVAILABLE = False
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print(f"[INTELLIGENT_NODES] ERROR - Import failed: {e}")
    print("[INTELLIGENT_NODES] Full traceback:")
    traceback.print_exc()
    print("[INTELLIGENT_NODES] Please install required dependencies: pip install optuna torch torchvision numpy pillow matplotlib")
except Exception as e:
    INTELLIGENT_AVAILABLE = False
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    print(f"[INTELLIGENT_NODES] ERROR - Unexpected error: {e}")
    print("[INTELLIGENT_NODES] Full traceback:")
    traceback.print_exc()

# Version info
__version__ = "3.0.0"
__author__ = "Intelligent Self-Optimizing Nodes for ComfyUI"

if INTELLIGENT_AVAILABLE:
    print(f"\n🧠 Intelligent Nodes v{__version__} loaded successfully")
    print(f"Total nodes available: {len(NODE_CLASS_MAPPINGS)}")
    
    # Check for optional dependencies
    print("\nDependency Status:")
    
    try:
        import optuna
        print("✓ Optuna (Core optimization engine)")
    except ImportError:
        print("✗ Optuna - REQUIRED! Install with: pip install optuna")
    
    try:
        import torch
        print("✓ PyTorch (Neural network backend)")
    except ImportError:
        print("✗ PyTorch - REQUIRED! Install with: pip install torch")
    
    try:
        import lpips
        print("✓ LPIPS (Perceptual similarity)")
    except ImportError:
        print("✗ LPIPS - Optional. Install with: pip install lpips")
    
    try:
        from dreamsim import dreamsim
        print("✓ DreamSim (Human perceptual similarity)")
    except ImportError:
        print("✗ DreamSim - Optional. Install with: pip install dreamsim")
    
    try:
        import matplotlib
        print("✓ Matplotlib (Visualization)")
    except ImportError:
        print("✗ Matplotlib - Optional. Install with: pip install matplotlib")
    
    print("\n💡 Quick Start:")
    print("1. Load 'intelligent_workflow_example.json'")
    print("2. Set your target image")
    print("3. Click 'Queue Prompt' to start learning")
    print("4. Each click improves the parameters!")
else:
    print("\n❌ Intelligent Nodes failed to load")
    print("Please check the error message above and install missing dependencies")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']