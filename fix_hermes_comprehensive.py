"""
Comprehensive fix for hermes.py syntax errors
"""

def fix_hermes_comprehensive():
    # Read the file
    with open('agents/hermes.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process line by line to fix syntax errors
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Fix missing commas in function calls
        if 'cast(AIMessage response)' in line:
            line = line.replace('cast(AIMessage response)', 'cast(AIMessage, response)')
        
        if 'cast(RunnableConfig config_with_limit)' in line:
            line = line.replace('cast(RunnableConfig config_with_limit)', 'cast(RunnableConfig, config_with_limit)')
        
        if 'cast(AIMessage prev_msg)' in line:
            line = line.replace('cast(AIMessage prev_msg)', 'cast(AIMessage, prev_msg)')
        
        # Fix getattr calls
        if 'getattr(' in line and '"' in line:
            import re
            line = re.sub(r'getattr\(([^,]+) "([^"]+)" ([^)]+)\)', r'getattr(\1, "\2", \3)', line)
        
        # Fix state.get calls
        if 'state.get(' in line:
            import re
            line = re.sub(r'state\.get\("([^"]+)" ([^)]+)\)', r'state.get("\1", \2)', line)
        
        # Fix isinstance calls
        if 'isinstance(' in line and ' dict)' in line:
            line = line.replace(' dict)', ', dict)')
        
        # Fix content[0].get calls
        if 'content[0].get(' in line:
            import re
            line = re.sub(r'content\[0\]\.get\("([^"]+)" "([^"]*)"\)', r'content[0].get("\1", "\2")', line)
        
        # Fix replace calls
        if '.replace(' in line:
            import re
            line = re.sub(r'replace\("([^"]+)" "([^"]*)"\)', r'replace("\1", "\2")', line)
        
        # Fix split calls
        if '.split(' in line:
            import re
            line = re.sub(r'split\("([^"]+)" ([^)]+)\)', r'split("\1", \2)', line)
        
        # Fix dictionary definitions
        if '{"configurable": {"thread_id": uuid} "recursion_limit": 66}' in line:
            line = line.replace('{"configurable": {"thread_id": uuid} "recursion_limit": 66}', 
                              '{"configurable": {"thread_id": uuid}, "recursion_limit": 66}')
        
        # Fix return statement
        if 'return graph.invoke(initial_state config=config_with_limit)' in line:
            line = line.replace('return graph.invoke(initial_state config=config_with_limit)', 
                              'return graph.invoke(initial_state, config=config_with_limit)')
        
        # Fix AIMessage constructor - look for the specific pattern
        if 'content=cleaned_content' in line and i + 1 < len(lines) and 'tool_calls=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        elif 'tool_calls=' in line and i + 1 < len(lines) and 'id=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        elif 'id=getattr(' in line and i + 1 < len(lines) and 'additional_kwargs=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        elif 'additional_kwargs=' in line and i + 1 < len(lines) and 'response_metadata=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        elif 'response_metadata=' in line and i + 1 < len(lines) and 'name=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        elif 'name=getattr(' in line and i + 1 < len(lines) and 'tool_call_chunks=' in lines[i + 1]:
            line = line.rstrip() + ',\n'
        
        fixed_lines.append(line)
        i += 1
    
    # Now add the tool_call_id fix
    content = ''.join(fixed_lines)
    
    # Find and replace the AIMessage creation with tool_call fix
    tool_fix_section = '''            ai_response = cast(AIMessage, response)
            
            # Fix tool calls to ensure they have proper IDs
            fixed_tool_calls = []
            if ai_response.tool_calls:
                import uuid as uuid_module
                for i, tool_call in enumerate(ai_response.tool_calls):
                    # Ensure tool_call is a dict and has an 'id' field
                    if isinstance(tool_call, dict):
                        if not tool_call.get("id"):
                            tool_call["id"] = f"call_{uuid_module.uuid4().hex[:8]}"
                        fixed_tool_calls.append(tool_call)
                    else:
                        # If tool_call is not a dict, try to convert it or create a proper structure
                        fixed_tool_call = {
                            "id": f"call_{uuid_module.uuid4().hex[:8]}",
                            "name": getattr(tool_call, "name", f"unknown_tool_{i}"),
                            "args": getattr(tool_call, "args", {})
                        }
                        fixed_tool_calls.append(fixed_tool_call)
            
            cleaned_response = AIMessage(
                content=cleaned_content,
                tool_calls=fixed_tool_calls,
                id=getattr(ai_response, "id", None),
                additional_kwargs=getattr(ai_response, "additional_kwargs", {}),
                response_metadata=getattr(ai_response, "response_metadata", {}),
                name=getattr(ai_response, "name", None),
                tool_call_chunks=getattr(
                    ai_response, "tool_call_chunks", None
                ),  # Preserve if exists
            )'''
    
    # Find the section to replace
    import re
    pattern = r'ai_response = cast\(AIMessage, response\).*?tool_call_chunks=getattr\(\s*ai_response, "tool_call_chunks", None\s*\),.*?\)'
    content = re.sub(pattern, tool_fix_section, content, flags=re.DOTALL)
    
    # Write the fixed file
    with open('agents/hermes.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Applied comprehensive fix to hermes.py")

if __name__ == "__main__":
    fix_hermes_comprehensive()