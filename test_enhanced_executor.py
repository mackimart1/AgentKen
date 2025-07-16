#!/usr/bin/env python3
"""Simple test of enhanced secure code executor."""

from tools.secure_code_executor_enhanced import secure_code_executor_enhanced
import json

# Test safe Python code
print("Testing safe Python code execution...")
result = secure_code_executor_enhanced.invoke({
    "code": "print('Hello from Enhanced Secure Code Executor!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')",
    "language": "python",
    "environment": "native",
    "max_execution_time": 10
})

data = json.loads(result)
print(f"Status: {data['status']}")
print(f"Output: {data.get('stdout', 'No output')}")
print(f"Execution time: {data.get('execution_time', 0):.2f} seconds")
print(f"Environment: {data.get('environment', 'unknown')}")

if data['status'] == 'success':
    print("✅ Enhanced Secure Code Executor working correctly!")
else:
    print(f"❌ Execution failed: {data.get('message', 'Unknown error')}")