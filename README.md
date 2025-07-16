# AgentKen

AgentKen is a self-evolving intelligence framework designed to build, expand, and optimize its own capabilities over time.
Starting from a minimal "kernel" of core agents, AgentKen dynamically creates new agents and tools to solve complex user tasks — growing its intelligence without human intervention.

✨ Core Concept
AgentKen begins as a minimal system of core agents, and bootstraps itself by:

- Decomposing tasks
- Creating new specialized agents and tools as needed
- Coordinating multiple agents to collaboratively solve problems
- Learning from its successes and failures over time

The ultimate goal is to create a flexible foundation for self-improving AGI architectures.

🛠 Architecture
**Agents (/agents)**
- Modular Python agents with specialized roles
- Core kernel agents include:
  - **Hermes** — Orchestrates tasks and agent collaboration
  - **AgentSmith** — Creates, tests, and manages new agents
  - **ToolMaker** — Designs and maintains new tools
  - **WebResearcher** — Gathers external information
- Additional agents (e.g., code_executor, ml_engineer, software_engineer, meta_agent) extend the system's reach into code execution, machine learning, and meta-reasoning.

**Tools (/tools)**
- Python modules that extend agent capabilities
- Examples:
  - Filesystem operations (read/write/list/delete)
  - Web search and scraping
  - Secure shell command and code execution
  - ML training and prediction
  - Human input requests
  - Text processing (e.g., Named Entity Recognition)

**Core Logic**
- Orchestration: Built on top of LangChain and LangGraph frameworks
- Execution: Dockerized environment for isolated, reproducible builds
- Memory Management: Integrated with ChromaDB for persistent vector storage (long-term memory)
- Configuration: Controlled through config.py and environment variables (.env)

⚡ Key Features
- **Self-evolving intelligence** — AgentKen can autonomously expand its capabilities without external programming
- **Dynamic agent and tool generation** — New components are created, tested, and deployed at runtime
- **Memory and state persistence** — Long-term storage of knowledge and skills using vector databases
- **Highly modular architecture** — Clean separation between agents, tools, memory, and orchestration
- **Test-driven development** — Extensive test coverage to ensure stability and reliability
- **Containerized deployment** — Fast setup via Docker and Docker Compose

🛠 How AgentKen Works
AgentKen is designed from the ground up to be a self-evolving AI framework.
It simulates the growth of intelligence by starting small and expanding its capabilities over time.

Here's a breakdown of the architecture and key ideas behind AgentKen:

🔹 **1. The Kernel Agents**
AgentKen begins with a small set of core agents, often called the kernel:

- **Hermes** — The master coordinator. It receives user tasks, decomposes them into sub-tasks, and assigns agents to solve them.
- **AgentSmith** — The agent creator. When new skills are needed, AgentSmith designs and deploys new agents.
- **ToolMaker** — The tool creator. When agents need new abilities, ToolMaker writes and integrates new Python tools.
- **WebResearcher** — The information gatherer. When external data or research is required, WebResearcher retrieves it.

The kernel is minimal but powerful — it gives AgentKen the ability to build itself over time.

🔹 **2. Agents and Collaboration**
Each agent is a modular Python module that:
- Has a clear role (e.g., executing code, training models, writing tools)
- Can request help from other agents
- Communicates through structured task requests
- Improves over time by using new tools and collaborating smarter

Agents don't just solve problems individually — they team up to tackle complex tasks.

🔹 **3. Tools and Skills**
AgentKen uses tools to give agents abilities they wouldn't otherwise have.

Examples:
- Reading and writing files
- Executing shell commands safely
- Searching the internet and scraping content
- Training machine learning models
- Performing Named Entity Recognition (NER)

When an agent needs to perform a task it can't handle yet, ToolMaker can design new tools automatically to fill the gap.

🔹 **4. Memory and Knowledge Retention**
AgentKen uses a vector database (ChromaDB) to store memories, which include:
- Information gathered from the web
- Past actions and outcomes
- New agent and tool designs
- Key learnings from completed tasks

This allows AgentKen to recall and learn from past experiences, making future problem-solving faster and smarter.

Memory is managed through the memory_manager.py system, supporting efficient storage and retrieval.

🔹 **5. Orchestration and Execution**
AgentKen uses:
- LangChain and LangGraph to build complex workflows and conversations between agents
- A central orchestration file (agent_kernel.py) to manage the life cycle of tasks and agent collaboration
- Docker containers to isolate the environment and ensure consistent execution
- Testing modules to maintain system stability and avoid regressions as the system grows

AgentKen is designed to observe, analyze, adapt, and grow — becoming more powerful with every cycle.

🌱 Core Philosophy
- **Self-Improvement First**: AgentKen should be able to expand its abilities without rewriting core code.
- **Minimal Seed, Infinite Growth**: Start small. Let the system bootstrap itself.
- **Agents as Experts**: Each agent should specialize and collaborate, not be monolithic.
- **Persistent Knowledge**: Learn from history. Don't reinvent solved problems.
- **Modularity**: Keep everything clean, swappable, and testable.

🧩 Why Build AgentKen This Way?
Building AGI (Artificial General Intelligence) is not about brute force.
It's about creating a living architecture that:
- Learns
- Evolves
- Repairs itself
- Grows intelligently over time

AgentKen is an early step toward systems that can program themselves, grow their own capabilities, and ultimately think and act with increasing autonomy.

✨ Summary
AgentKen is more than just an AI framework.
It's a self-building, self-expanding, and self-evolving system — built with the dream of creating a new kind of intelligence, one step at a time.
