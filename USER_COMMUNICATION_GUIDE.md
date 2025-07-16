# Inferra V Enhanced System - User Communication Guide

## 🎯 **Overview**

The Inferra V Enhanced System now provides multiple ways for users to communicate with and control the agent system. This guide covers all available communication methods and how to use them effectively.

## 🚀 **Available Communication Methods**

### 1. **Interactive CLI Interface** (`user_interface.py`)
A command-line interface for direct interaction with the system.

### 2. **Web API Interface** (`web_api.py`)
RESTful API for external applications and web interfaces.

### 3. **Python Client** (`client_example.py`)
Programmatic access for Python applications.

### 4. **Direct System Integration**
Direct integration with the core system components.

---

## 📱 **Method 1: Interactive CLI Interface**

### **Starting the Interface**
```bash
python user_interface.py
```

### **Available Commands**
```
🔧 AVAILABLE COMMANDS
==================================================
🔹 init                  - Initialize the Inferra V system
🔹 status               - Show system status and health
🔹 create_plan          - Create a new execution plan
🔹 execute_plan <id>    - Execute a specific plan
🔹 monitor_plan <id>    - Monitor plan execution
🔹 list_plans           - List all available plans
🔹 chat                 - Start chat mode with the agent
🔹 quick_start          - Run a demonstration
🔹 optimize             - Get optimization recommendations
🔹 shutdown             - Shutdown the system
🔹 exit                 - Exit the interface
```

### **Example Session**
```
🤖 Inferra> init
🔄 Initializing Inferra V Enhanced System...
✅ System initialized successfully!

🤖 Inferra> status
📊 SYSTEM STATUS
🟢 System Health: 96.2/100
📋 Active Plans: 0
🤖 Agents: 1 active

🤖 Inferra> quick_start
🚀 QUICK START DEMO
✅ Created plan: Market Analysis Project
🚀 Executing plan...
📊 Progress: 100% (3/3 tasks)
🎉 Demo completed!

🤖 Inferra> chat
💬 CHAT MODE
You: What is the system status?
🤖 Inferra: System health is 96.2/100. All systems operational!

🤖 Inferra> exit
👋 Goodbye!
```

---

## 🌐 **Method 2: Web API Interface**

### **Starting the API Server**
```bash
python web_api.py
```

The API will be available at: `http://localhost:5000`

### **API Endpoints**

#### **System Management**
- `GET /api/status` - Get system status
- `POST /api/initialize` - Initialize system
- `POST /api/shutdown` - Shutdown system

#### **Plan Management**
- `GET /api/plans` - List all plans
- `POST /api/plans` - Create new plan
- `GET /api/plans/{id}` - Get plan details
- `POST /api/plans/{id}/execute` - Execute plan
- `GET /api/plans/{id}/status` - Get plan status

#### **System Information**
- `GET /api/agents` - List agents
- `GET /api/tools` - List tools
- `GET /api/metrics` - Get metrics
- `GET /api/alerts` - Get alerts
- `GET /api/optimize` - Get optimization recommendations

#### **Communication**
- `POST /api/chat` - Chat with system

### **Example API Usage**

#### **Initialize System**
```bash
curl -X POST http://localhost:5000/api/initialize
```

#### **Get System Status**
```bash
curl http://localhost:5000/api/status
```

#### **Create a Plan**
```bash
curl -X POST http://localhost:5000/api/plans \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Plan",
    "description": "Test plan",
    "tasks": [
      {
        "id": "task1",
        "name": "Search Task",
        "agent_type": "research_agent",
        "capability": "web_search",
        "parameters": {"query": "AI trends", "max_results": 5},
        "priority": 3,
        "estimated_duration": 10.0,
        "dependencies": [],
        "required_resources": ["internet_access"]
      }
    ]
  }'
```

#### **Chat with System**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the system health?"}'
```

---

## 🐍 **Method 3: Python Client**

### **Basic Usage**
```python
from client_example import InferraVClient

# Create client
client = InferraVClient("http://localhost:5000")

# Initialize system
client.initialize_system()

# Check status
status = client.get_status()
print(f"Health: {status['data']['health_score']}/100")

# Create and execute plan
plan_id = client.create_plan(
    name="My Plan",
    description="Test plan",
    tasks=[...]  # Task definitions
)

client.execute_plan(plan_id)
final_status = client.monitor_plan(plan_id)

# Chat with system
response = client.chat("How are you doing?")
print(response['data']['response'])
```

### **Running Examples**
```bash
# Basic demo
python client_example.py

# Advanced demo
python client_example.py advanced

# Interactive mode
python client_example.py interactive
```

---

## 🔧 **Method 4: Direct Integration**

### **Direct System Access**
```python
from core.workflow_monitoring import WorkflowMonitor
from core.tool_integration_system import create_tool_system
from core.agent_framework import create_agent_system
from core.adaptive_orchestrator import AdvancedOrchestrator

# Initialize components
monitor = WorkflowMonitor()
monitor.start()

tool_registry = create_tool_system()
message_bus, _, agents = create_agent_system()
orchestrator = AdvancedOrchestrator(message_bus, tool_registry)

# Use the system directly
plan_id = orchestrator.create_plan(...)
orchestrator.execute_plan(plan_id)
```

---

## 💬 **Communication Features**

### **Chat Capabilities**
The system supports natural language communication:

- **Status Queries**: "What is the system status?"
- **Plan Management**: "Create a new plan for data analysis"
- **Help Requests**: "What can you do?"
- **Optimization**: "How can I improve performance?"

### **Real-time Monitoring**
- Live plan execution tracking
- System health monitoring
- Resource utilization alerts
- Performance optimization suggestions

### **Interactive Features**
- Step-by-step plan creation
- Guided task configuration
- Real-time feedback
- Error handling and recovery

---

## 🎯 **Use Cases**

### **1. Development and Testing**
```bash
# Use CLI for quick testing
python user_interface.py
🤖 Inferra> quick_start
```

### **2. Web Application Integration**
```javascript
// Use API from web applications
fetch('/api/status')
  .then(response => response.json())
  .then(data => console.log(data));
```

### **3. Automated Scripts**
```python
# Use Python client for automation
client = InferraVClient()
client.initialize_system()
plan_id = client.create_plan(...)
client.execute_plan(plan_id)
```

### **4. Custom Applications**
```python
# Direct integration for custom apps
orchestrator = AdvancedOrchestrator(...)
# Custom logic here
```

---

## 🚀 **Getting Started**

### **Quick Start (Recommended)**
1. **Start the CLI interface**:
   ```bash
   python user_interface.py
   ```

2. **Initialize the system**:
   ```
   🤖 Inferra> init
   ```

3. **Run a demo**:
   ```
   🤖 Inferra> quick_start
   ```

4. **Try chat mode**:
   ```
   🤖 Inferra> chat
   You: Hello, how are you?
   ```

### **API Development**
1. **Start the API server**:
   ```bash
   python web_api.py
   ```

2. **Visit the documentation**:
   Open `http://localhost:5000` in your browser

3. **Test with curl**:
   ```bash
   curl -X POST http://localhost:5000/api/initialize
   ```

### **Python Integration**
1. **Run the client example**:
   ```bash
   python client_example.py
   ```

2. **Try interactive mode**:
   ```bash
   python client_example.py interactive
   ```

---

## 🔍 **Troubleshooting**

### **Common Issues**

#### **"System not initialized"**
- **Solution**: Run `init` command or call `/api/initialize`

#### **"Flask not available"**
- **Solution**: Install Flask: `pip install flask flask-cors`

#### **"Connection refused"**
- **Solution**: Make sure API server is running: `python web_api.py`

#### **"Plan not found"**
- **Solution**: Use `list_plans` to see available plans

### **Debug Mode**
Enable debug logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 **Performance Tips**

### **Optimal Usage Patterns**
1. **Initialize once**: Keep the system initialized for multiple operations
2. **Batch operations**: Create multiple plans before executing
3. **Monitor efficiently**: Use appropriate polling intervals
4. **Resource management**: Monitor resource utilization

### **Best Practices**
- Use appropriate task priorities
- Set realistic duration estimates
- Monitor system health regularly
- Handle errors gracefully
- Use chat for quick status checks

---

## 🎉 **Conclusion**

The Inferra V Enhanced System now provides comprehensive user communication capabilities through multiple interfaces:

- **CLI Interface**: Perfect for development and testing
- **Web API**: Ideal for web applications and external integrations
- **Python Client**: Great for automation and scripting
- **Direct Integration**: Maximum flexibility for custom applications

Choose the method that best fits your use case and start communicating with your enhanced agent system today!

**Happy orchestrating! 🚀**