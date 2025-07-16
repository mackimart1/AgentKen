#!/usr/bin/env python3
"""
Script to fix import issues in meta_agent.py
"""


def fix_meta_agent_imports():
    """Fix import issues in meta_agent.py"""
    file_path = "agents/meta_agent.py"

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix the import
    content = content.replace(
        "from .reinforcement_learning import ReinforcementLearningManager",
        "from .reinforcement_learning import EnhancedReinforcementLearningManager as ReinforcementLearningManager",
    )

    # Write the file back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Fixed meta_agent.py imports")


if __name__ == "__main__":
    fix_meta_agent_imports()
