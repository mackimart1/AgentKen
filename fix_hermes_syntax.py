"""
Fix syntax errors in hermes.py by adding missing commas and fixing tool_call_id issue
"""

import re

def fix_hermes_syntax():
    # Read the file
    with open('agents/hermes.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix missing commas in function calls and parameters
    fixes = [
        # Fix cast function call
        (r'cast\(AIMessage response\)', r'cast(AIMessage, response)'),
        
        # Fix getattr calls
        (r'getattr\(([^,]+) "([^"]+)" ([^)]+)\)', r'getattr(\1, "\2", \3)'),
        
        # Fix AIMessage constructor parameters
        (r'content=cleaned_content\n\s+tool_calls=', r'content=cleaned_content,\n                tool_calls='),
        (r'tool_calls=([^\n]+)\n\s+id=', r'tool_calls=\1,\n                id='),
        (r'id=([^\n]+)\n\s+additional_kwargs=', r'id=\1,\n                additional_kwargs='),
        (r'additional_kwargs=([^\n]+)\n\s+response_metadata=', r'additional_kwargs=\1,\n                response_metadata='),
        (r'response_metadata=([^\n]+)\n\s+name=', r'response_metadata=\1,\n                name='),
        (r'name=([^\n]+)\n\s+tool_call_chunks=', r'name=\1,\n                tool_call_chunks='),
        
        # Fix other function calls
        (r'state\.get\("([^"]+)" ([^)]+)\)', r'state.get("\1", \2)'),
        (r'isinstance\(([^,]+) dict\)', r'isinstance(\1, dict)'),
        (r'content\[0\]\.get\("([^"]+)" "([^"]*)"\)', r'content[0].get("\1", "\2")'),
        (r'replace\("([^"]+)" "([^"]*)"\)', r'replace("\1", "\2")'),
        (r'split\("([^"]+)" ([^)]+)\)', r'split("\1", \2)'),
        
        # Fix dictionary access
        (r'{"configurable": {"thread_id": uuid} "recursion_limit": 66}', 
         r'{"configurable": {"thread_id": uuid}, "recursion_limit": 66}'),
        
        # Fix cast calls
        (r'cast\(RunnableConfig config_with_limit\)', r'cast(RunnableConfig, config_with_limit)'),
        (r'cast\(AIMessage prev_msg\)', r'cast(AIMessage, prev_msg)'),
        
        # Fix return statements
        (r'return graph\.invoke\(initial_state config=config_with_limit\)', 
         r'return graph.invoke(initial_state, config=config_with_limit)'),
    ]
    
    # Apply fixes
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Add the tool_call_id fix
    tool_fix = '''
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
                tool_calls=fixed_tool_calls,'''
    
    # Replace the tool_calls assignment
    content = re.sub(
        r'cleaned_response = AIMessage\(\s*content=cleaned_content,\s*tool_calls=ai_response\.tool_calls,',
        tool_fix,
        content
    )
    
    # Write the fixed file
    with open('agents/hermes.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed hermes.py syntax errors and tool_call_id issue")

if __name__ == "__main__":
    fix_hermes_syntax()