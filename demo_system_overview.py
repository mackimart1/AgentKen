#!/usr/bin/env python3
"""
Demonstration script showing how Hermes can now view the AgentK system
using the new detailed listing and overview tools.
"""

from tools.list_agents_detailed import list_agents_detailed
from tools.list_tools_detailed import list_tools_detailed
from tools.system_overview import system_overview

def demo_system_overview():
    """Demonstrate the new system overview capabilities."""
    
    print("=" * 80)
    print("AGENTK SYSTEM OVERVIEW DEMONSTRATION")
    print("=" * 80)
    print()
    
    # 1. System Overview (Summary)
    print("1. SYSTEM OVERVIEW (SUMMARY)")
    print("-" * 40)
    result = system_overview.invoke({"format_output": "summary"})
    print(result)
    print()
    
    # 2. Detailed Agent Listing (Summary)
    print("2. AGENTS OVERVIEW (SUMMARY)")
    print("-" * 40)
    result = list_agents_detailed.invoke({"format_output": "summary"})
    print(result)
    print()
    
    # 3. Detailed Tools Listing (Summary)
    print("3. TOOLS OVERVIEW (SUMMARY)")
    print("-" * 40)
    result = list_tools_detailed.invoke({"format_output": "summary"})
    print(result)
    print()
    
    # 4. Filter agents by capability
    print("4. AGENTS WITH CODE DEVELOPMENT CAPABILITY")
    print("-" * 40)
    result = list_agents_detailed.invoke({
        "filter_by_capability": "code_development",
        "format_output": "detailed"
    })
    print(result)
    print()
    
    # 5. Show one detailed agent
    print("5. DETAILED VIEW OF SOFTWARE ENGINEER")
    print("-" * 40)
    # This would show all agents, but we can see the software_engineer in the output
    result = list_agents_detailed.invoke({
        "format_output": "detailed"
    })
    # Extract just the software_engineer part (simplified for demo)
    lines = result.split('\n')
    in_software_engineer = False
    software_engineer_lines = []
    
    for line in lines:
        if 'SOFTWARE_ENGINEER' in line.upper():
            in_software_engineer = True
        elif in_software_engineer and line.startswith('='):
            break
        
        if in_software_engineer:
            software_engineer_lines.append(line)
    
    if software_engineer_lines:
        print('\n'.join(software_engineer_lines))
    else:
        print("Software Engineer details not found in output")
    
    print()
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    demo_system_overview()