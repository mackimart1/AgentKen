#!/usr/bin/env python3
"""
Fix Manifest Entries Script

This script fixes existing manifest entries that are missing required fields
like module_path and function_name. It updates both agents_manifest.json and
tools_manifest.json to ensure all entries have the proper structure.
"""
import json
import os
import sys
from typing import Dict, Any, List


def fix_manifest_entries(manifest_file: str, entity_type: str) -> bool:
    """
    Fix manifest entries in the specified file.

    Args:
        manifest_file: Path to the manifest file
        entity_type: Either 'agents' or 'tools'

    Returns:
        bool: True if fixes were applied, False otherwise
    """
    if not os.path.exists(manifest_file):
        print(f"Manifest file {manifest_file} does not exist. Skipping.")
        return False

    try:
        # Load the manifest
        with open(manifest_file, "r") as f:
            manifest_data = json.load(f)

        # Handle both list and dict formats
        if isinstance(manifest_data, list):
            entries = manifest_data
        else:
            entries = manifest_data.get(entity_type, [])

        fixed_count = 0
        fixed_entries = []

        for entry in entries:
            if not isinstance(entry, dict):
                print(f"Warning: Skipping non-dict entry: {entry}")
                fixed_entries.append(entry)
                continue

            # Check if entry is missing required fields
            needs_fix = False

            if entity_type == "tools":
                # For tools, check for module_path and function_name
                if "module_path" not in entry or "function_name" not in entry:
                    needs_fix = True
                    if "name" in entry:
                        tool_filename = entry["name"]
                        entry["module_path"] = f"tools/{tool_filename}.py"
                        entry["function_name"] = tool_filename
                        fixed_count += 1
                        print(f"Fixed tool entry: {tool_filename}")

            elif entity_type == "agents":
                # For agents, check for module_path and function_name
                if "module_path" not in entry or "function_name" not in entry:
                    needs_fix = True
                    if "name" in entry:
                        agent_filename = entry["name"]
                        entry["module_path"] = f"agents/{agent_filename}.py"
                        entry["function_name"] = agent_filename
                        fixed_count += 1
                        print(f"Fixed agent entry: {agent_filename}")

            fixed_entries.append(entry)

        if fixed_count > 0:
            # Write back the fixed manifest
            if isinstance(manifest_data, list):
                # Keep as list format
                with open(manifest_file, "w") as f:
                    json.dump(fixed_entries, f, indent=2)
            else:
                # Keep as dict format
                manifest_data[entity_type] = fixed_entries
                with open(manifest_file, "w") as f:
                    json.dump(manifest_data, f, indent=2)

            print(f"Fixed {fixed_count} {entity_type} entries in {manifest_file}")
            return True
        else:
            print(f"No fixes needed for {entity_type} in {manifest_file}")
            return False

    except Exception as e:
        print(f"Error fixing {manifest_file}: {e}")
        return False


def main():
    """Main function to fix all manifest files"""
    print("🔧 Fixing Manifest Entries")
    print("=" * 40)

    # Fix tools manifest
    tools_fixed = fix_manifest_entries("tools_manifest.json", "tools")

    # Fix agents manifest
    agents_fixed = fix_manifest_entries("agents_manifest.json", "agents")

    print("\n" + "=" * 40)
    if tools_fixed or agents_fixed:
        print("✅ Manifest fixes completed successfully!")
        print("📝 All entries now have required module_path and function_name fields.")
    else:
        print("ℹ️  No manifest fixes were needed.")

    print(
        "\n🚀 You can now restart your project and the manifest errors should be resolved."
    )


if __name__ == "__main__":
    main()
