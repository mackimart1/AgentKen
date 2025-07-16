#!/usr/bin/env python3
"""
Script to register LogAnalyzer tools in the tools manifest.
"""

import json
import os
from datetime import datetime

def register_log_analyzer_tools():
    """Register LogAnalyzer tools in the tools manifest."""
    
    # Define the LogAnalyzer tools to register
    log_analyzer_tools = [
        {
            "name": "analyze_log_file",
            "module_path": "tools/log_analyzer.py",
            "function_name": "analyze_log_file",
            "description": "Analyze a log file and return comprehensive analysis results including error patterns, anomalies, and recommendations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the log file to analyze"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to analyze (default: 1000)",
                        "default": 1000
                    }
                },
                "required": ["file_path"]
            },
            "output_schema": {
                "type": "string",
                "description": "JSON string containing detailed log analysis results"
            },
            "author": "LogAnalyzer System",
            "created_at": "2025-01-16T00:00:00.000000",
            "version": "1.0.0",
            "status": "active"
        },
        {
            "name": "analyze_log_content",
            "module_path": "tools/log_analyzer.py",
            "function_name": "analyze_log_content",
            "description": "Analyze log content directly and return comprehensive analysis results including error detection, pattern recognition, and performance metrics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "Raw log content to analyze"
                    }
                },
                "required": ["log_content"]
            },
            "output_schema": {
                "type": "string",
                "description": "JSON string containing detailed log analysis results"
            },
            "author": "LogAnalyzer System",
            "created_at": "2025-01-16T00:00:00.000000",
            "version": "1.0.0",
            "status": "active"
        },
        {
            "name": "find_log_files",
            "module_path": "tools/log_analyzer.py",
            "function_name": "find_log_files",
            "description": "Find log files in a directory matching a pattern with file metadata and accessibility information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for log files"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "File pattern to match (default: '*.log')",
                        "default": "*.log"
                    }
                },
                "required": ["directory"]
            },
            "output_schema": {
                "type": "string",
                "description": "JSON string containing list of found log files with metadata"
            },
            "author": "LogAnalyzer System",
            "created_at": "2025-01-16T00:00:00.000000",
            "version": "1.0.0",
            "status": "active"
        },
        {
            "name": "analyze_error_patterns",
            "module_path": "tools/log_analyzer.py",
            "function_name": "analyze_error_patterns",
            "description": "Analyze error patterns in log content and identify recurring issues with pattern frequency and examples.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "Raw log content to analyze"
                    },
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum occurrences to consider a pattern (default: 2)",
                        "default": 2
                    }
                },
                "required": ["log_content"]
            },
            "output_schema": {
                "type": "string",
                "description": "JSON string containing error pattern analysis with frequency and examples"
            },
            "author": "LogAnalyzer System",
            "created_at": "2025-01-16T00:00:00.000000",
            "version": "1.0.0",
            "status": "active"
        }
    ]
    
    # Load existing tools manifest
    manifest_path = "tools_manifest.json"
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            existing_tools = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {manifest_path} not found, creating new manifest")
        existing_tools = []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {manifest_path}: {e}")
        return False
    
    # Check if tools are already registered
    existing_tool_names = {tool.get('name') for tool in existing_tools}
    new_tools = []
    
    for tool in log_analyzer_tools:
        if tool['name'] not in existing_tool_names:
            new_tools.append(tool)
            print(f"Adding new tool: {tool['name']}")
        else:
            print(f"Tool already exists: {tool['name']}")
    
    if new_tools:
        # Add new tools to the manifest
        existing_tools.extend(new_tools)
        
        # Write updated manifest
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(existing_tools, f, indent=4, ensure_ascii=False)
            
            print(f"\n✅ Successfully registered {len(new_tools)} LogAnalyzer tools!")
            print(f"Updated manifest saved to: {manifest_path}")
            
            # Print summary
            print(f"\nRegistered tools:")
            for tool in new_tools:
                print(f"  - {tool['name']}: {tool['description'][:60]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing manifest: {e}")
            return False
    else:
        print("✅ All LogAnalyzer tools are already registered!")
        return True


def main():
    """Main function."""
    print("Registering LogAnalyzer tools in the tools manifest...")
    print(f"Started at: {datetime.now()}")
    
    success = register_log_analyzer_tools()
    
    if success:
        print(f"\n🎉 LogAnalyzer tools registration completed successfully!")
    else:
        print(f"\n⚠️ LogAnalyzer tools registration failed!")
    
    print(f"Completed at: {datetime.now()}")
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)