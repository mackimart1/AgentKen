#!/usr/bin/env python3
"""
Cleanup script to remove temporary files created during integration testing.
"""

import os
import glob

def cleanup_integration_files():
    """Remove temporary files created during integration."""
    
    files_to_remove = [
        "fix_meta_agent.py",
        "fix_meta_agent_imports.py",
        "test_openrouter.py",
        "test_full_system_integration.py",
        "cleanup_integration_files.py"  # This script itself
    ]
    
    # Remove specific files
    removed_files = []
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"✅ Removed: {file_path}")
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
    
    # Remove any __pycache__ directories
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    for cache_dir in pycache_dirs:
        try:
            import shutil
            shutil.rmtree(cache_dir)
            removed_files.append(cache_dir)
            print(f"✅ Removed cache: {cache_dir}")
        except Exception as e:
            print(f"❌ Failed to remove {cache_dir}: {e}")
    
    # Remove any .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            removed_files.append(pyc_file)
            print(f"✅ Removed: {pyc_file}")
        except Exception as e:
            print(f"❌ Failed to remove {pyc_file}: {e}")
    
    print(f"\n🧹 Cleanup complete! Removed {len(removed_files)} files/directories.")
    print("\n📋 Integration files that remain:")
    print("  - SYSTEM_INTEGRATION_REPORT.md (keep for documentation)")
    print("  - All core system files (operational)")
    print("  - Configuration files (.env, manifests)")
    
    return len(removed_files)

if __name__ == "__main__":
    print("🧹 Starting integration file cleanup...")
    print("=" * 50)
    cleanup_integration_files()
    print("=" * 50)
    print("✅ Cleanup completed successfully!")