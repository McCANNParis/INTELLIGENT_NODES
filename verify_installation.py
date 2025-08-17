#!/usr/bin/env python3
"""
Verification script for Intelligent Nodes installation
Run this to check if all dependencies are installed correctly
"""

import sys
import os

print("=" * 60)
print("INTELLIGENT NODES - Installation Verification")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 7):
    print("   ❌ Python 3.7+ required")
else:
    print("   ✅ Python version OK")

# Check required dependencies
print("\n2. Required Dependencies:")

required_packages = {
    "torch": "PyTorch",
    "torchvision": "TorchVision", 
    "numpy": "NumPy",
    "PIL": "Pillow",
    "optuna": "Optuna"
}

all_required_ok = True
for package, name in required_packages.items():
    try:
        if package == "PIL":
            import PIL
            print(f"   ✅ {name} ({PIL.__version__})")
        else:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"   ✅ {name} ({version})")
    except ImportError:
        print(f"   ❌ {name} - NOT INSTALLED")
        print(f"      Install with: pip install {package if package != 'PIL' else 'Pillow'}")
        all_required_ok = False

# Check optional dependencies
print("\n3. Optional Dependencies:")

optional_packages = {
    "matplotlib": "Matplotlib (for visualization)",
    "lpips": "LPIPS (for perceptual similarity)",
    "dreamsim": "DreamSim (for human perceptual similarity)",
    "clip": "CLIP (for semantic similarity)"
}

for package, description in optional_packages.items():
    try:
        if package == "dreamsim":
            from dreamsim import dreamsim
            print(f"   ✅ {description}")
        elif package == "clip":
            import clip
            print(f"   ✅ {description}")
        else:
            mod = __import__(package)
            print(f"   ✅ {description}")
    except ImportError:
        print(f"   ⚠️  {description} - Not installed (optional)")

# Check if nodes can be imported
print("\n4. Node Import Test:")
try:
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from intelligent_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    print(f"   ✅ Successfully imported {len(NODE_CLASS_MAPPINGS)} nodes:")
    for node_name in NODE_CLASS_MAPPINGS.keys():
        print(f"      - {node_name}")
    
    # Test node instantiation
    print("\n5. Node Instantiation Test:")
    test_passed = True
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        try:
            # Check if the class has required methods
            if not hasattr(node_class, 'INPUT_TYPES'):
                print(f"   ❌ {node_name} missing INPUT_TYPES")
                test_passed = False
            if not hasattr(node_class, 'FUNCTION'):
                print(f"   ❌ {node_name} missing FUNCTION") 
                test_passed = False
        except Exception as e:
            print(f"   ❌ {node_name} failed: {e}")
            test_passed = False
    
    if test_passed:
        print("   ✅ All nodes have required methods")
        
except ImportError as e:
    print(f"   ❌ Failed to import nodes: {e}")
    print("\n   Troubleshooting:")
    print("   1. Make sure you're in the NODES_BAYESIAN directory")
    print("   2. Check that intelligent_nodes.py exists")
    print("   3. Install missing dependencies listed above")
except Exception as e:
    print(f"   ❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

# Check ComfyUI integration
print("\n6. ComfyUI Integration:")
comfyui_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if os.path.exists(os.path.join(comfyui_path, 'main.py')):
    print(f"   ✅ Appears to be in ComfyUI custom_nodes directory")
    print(f"      Path: {comfyui_path}")
else:
    print(f"   ⚠️  Not in standard ComfyUI structure")
    print(f"      Expected ComfyUI at: {comfyui_path}")
    print(f"      Make sure this folder is in: ComfyUI/custom_nodes/")

# Summary
print("\n" + "=" * 60)
if all_required_ok:
    print("✅ INSTALLATION COMPLETE - All required dependencies OK")
    print("\nNext steps:")
    print("1. Restart ComfyUI if it's running")
    print("2. Load test_workflow_minimal.json")
    print("3. Click 'Queue Prompt' to test")
else:
    print("❌ INSTALLATION INCOMPLETE - Missing required dependencies")
    print("\nPlease install missing packages and run this script again")

print("=" * 60)