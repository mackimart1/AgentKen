"""
Meta Agent: Enhanced self-reflection and analysis agent for the AgentK system.

Responsible for analyzing AgentK's performance, structure, and identifying
areas for improvement based on logs, memory, and system state.
"""

from typing import Literal, Dict, List, Any, Optional, Tuple
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# OpenRouter uses standard HTTP errors, not Google-specific exceptions
from requests.exceptions import HTTPError, RequestException
import logging

import utils
import config
import memory_manager
from .reinforcement_learning import (
    EnhancedReinforcementLearningManager as ReinforcementLearningManager,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetrics:
    """Container for system analysis metrics"""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    agent_utilization: Dict[str, int] = None
    tool_usage: Dict[str, int] = None
    error_patterns: Dict[str, int] = None
    performance_trends: List[float] = None

    def __post_init__(self):
        if self.agent_utilization is None:
            self.agent_utilization = {}
        if self.tool_usage is None:
            self.tool_usage = {}
        if self.error_patterns is None:
            self.error_patterns = {}
        if self.performance_trends is None:
            self.performance_trends = []

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate


class EnhancedReinforcementLearningManager:
    """Enhanced RL manager with better state representation and action selection"""

    def __init__(self):
        self.rl_manager = ReinforcementLearningManager()
        self.action_history = []
        self.reward_history = []
        self.state_history = []

    def create_enhanced_state(
        self,
        current_messages: List,
        relevant_memories: List,
        analysis_metrics: AnalysisMetrics,
        system_load: float = 0.0,
    ) -> np.ndarray:
        """Create enhanced state vector for better RL decision making"""

        state_features = [
            float(len(relevant_memories) > 0),  # Memory availability
            float(len(current_messages)),  # Message history length
            float(
                any(
                    isinstance(msg, AIMessage) and msg.tool_calls
                    for msg in current_messages
                )
            ),  # Tool usage history
            analysis_metrics.success_rate,  # Historical success rate
            analysis_metrics.failure_rate,  # Historical failure rate
            float(len(analysis_metrics.error_patterns)),  # Error diversity
            system_load,  # Current system load
            float(len(self.action_history)),  # Action history length
        ]

        return np.array(state_features, dtype=np.float32)

    def get_contextual_action(
        self, state_vector: np.ndarray, context: str = ""
    ) -> Tuple[int, str]:
        """Get action with contextual reasoning"""
        base_action = self.rl_manager.predict_action(state_vector)

        # Add contextual logic
        reasoning = ""
        if (
            "error" in context.lower() and base_action == 0
        ):  # If errors detected, consider retry
            if len(self.action_history) < 3:  # Avoid infinite retries
                base_action = 1
                reasoning = "Switching to retry due to error context"
        elif "timeout" in context.lower():
            base_action = 2
            reasoning = "Aborting due to timeout"
        elif state_vector[0] == 0 and base_action == 0:  # No memories available
            reasoning = "Continuing despite no memory context"

        self.action_history.append(base_action)
        return base_action, reasoning


class MetaAnalyzer:
    """Enhanced analysis engine for system introspection"""

    def __init__(self):
        self.metrics = AnalysisMetrics()
        self.analysis_cache = {}
        self.last_analysis_time = None

    def should_use_cached_analysis(self, task: str, max_age_minutes: int = 30) -> bool:
        """Check if we can use cached analysis results"""
        cache_key = hash(task)
        if cache_key in self.analysis_cache:
            cached_time, _ = self.analysis_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=max_age_minutes):
                return True
        return False

    def get_cached_analysis(self, task: str) -> Optional[str]:
        """Retrieve cached analysis if available"""
        cache_key = hash(task)
        if cache_key in self.analysis_cache:
            _, analysis = self.analysis_cache[cache_key]
            return analysis
        return None

    def cache_analysis(self, task: str, analysis: str):
        """Cache analysis results"""
        cache_key = hash(task)
        self.analysis_cache[cache_key] = (datetime.now(), analysis)

    def extract_metrics_from_logs(self, log_data: str) -> AnalysisMetrics:
        """Extract performance metrics from log data"""
        metrics = AnalysisMetrics()

        try:
            lines = log_data.split("\n")
            for line in lines:
                if "task completed" in line.lower():
                    metrics.successful_tasks += 1
                elif "task failed" in line.lower() or "error" in line.lower():
                    metrics.failed_tasks += 1
                elif "agent:" in line.lower():
                    agent_name = self._extract_agent_name(line)
                    if agent_name:
                        metrics.agent_utilization[agent_name] = (
                            metrics.agent_utilization.get(agent_name, 0) + 1
                        )
                elif "tool:" in line.lower():
                    tool_name = self._extract_tool_name(line)
                    if tool_name:
                        metrics.tool_usage[tool_name] = (
                            metrics.tool_usage.get(tool_name, 0) + 1
                        )

            metrics.total_tasks = metrics.successful_tasks + metrics.failed_tasks

        except Exception as e:
            logger.warning(f"Error extracting metrics from logs: {e}")

        return metrics

    def _extract_agent_name(self, line: str) -> Optional[str]:
        """Extract agent name from log line"""
        try:
            if "agent:" in line.lower():
                parts = line.split("agent:")
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
        except:
            pass
        return None

    def _extract_tool_name(self, line: str) -> Optional[str]:
        """Extract tool name from log line"""
        try:
            if "tool:" in line.lower():
                parts = line.split("tool:")
                if len(parts) > 1:
                    return parts[1].strip().split()[0]
        except:
            pass
        return None

    def generate_recommendations(self, metrics: AnalysisMetrics) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []

        # Performance-based recommendations
        if metrics.success_rate < 0.7:
            recommendations.append(
                f"Low success rate ({metrics.success_rate:.2%}). Consider reviewing agent prompts and tool effectiveness."
            )

        # Agent utilization recommendations
        if metrics.agent_utilization:
            underused_agents = [
                agent for agent, count in metrics.agent_utilization.items() if count < 2
            ]
            if underused_agents:
                recommendations.append(
                    f"Underutilized agents detected: {', '.join(underused_agents)}. Consider task routing optimization."
                )

        # Tool usage recommendations
        if metrics.tool_usage:
            unused_tools = self._identify_unused_tools(metrics.tool_usage)
            if unused_tools:
                recommendations.append(
                    f"Unused tools detected: {', '.join(unused_tools)}. Consider tool cleanup or better integration."
                )

        # Error pattern recommendations
        if metrics.error_patterns:
            frequent_errors = [
                (error, count)
                for error, count in metrics.error_patterns.items()
                if count > 5
            ]
            if frequent_errors:
                recommendations.append(
                    f"Frequent error patterns: {', '.join([f'{error} ({count}x)' for error, count in frequent_errors])}"
                )

        return recommendations

    def _identify_unused_tools(self, tool_usage: Dict[str, int]) -> List[str]:
        """Identify tools that are available but unused"""
        try:
            all_tools = utils.all_tool_functions()
            all_tool_names = [
                getattr(tool, "__name__", str(tool)) for tool in all_tools
            ]
            used_tool_names = set(tool_usage.keys())
            return [tool for tool in all_tool_names if tool not in used_tool_names]
        except:
            return []


# Initialize enhanced components
rl_manager = EnhancedReinforcementLearningManager()
analyzer = MetaAnalyzer()

# Enhanced system prompt with better structure
ENHANCED_SYSTEM_PROMPT = """You are the Meta-Agent, an advanced self-reflection and analysis system for AgentK. Your mission is to provide deep insights into system performance, identify optimization opportunities, and recommend concrete improvements.

**Core Responsibilities:**
1. **System Health Assessment**: Evaluate overall system performance, agent effectiveness, and resource utilization
2. **Pattern Recognition**: Identify recurring issues, successful patterns, and optimization opportunities  
3. **Predictive Analysis**: Forecast potential issues and recommend preventive measures
4. **Strategic Recommendations**: Propose architectural improvements and process optimizations

**Analysis Framework:**
1. **Data Collection Phase**: 
   - Gather comprehensive system data (logs, metrics, memory, agent states)
   - Validate data quality and completeness
   - Identify data gaps and limitations

2. **Analysis Phase**:
   - Performance trend analysis
   - Agent utilization patterns
   - Tool effectiveness assessment
   - Error pattern recognition
   - Resource optimization opportunities

3. **Synthesis Phase**:
   - Correlate findings across different data sources
   - Prioritize issues by impact and feasibility
   - Develop actionable recommendations
   - Estimate implementation effort and expected benefits

4. **Action Planning Phase**:
   - Design specific improvement tasks
   - Identify responsible agents for implementation
   - Set success criteria and monitoring mechanisms

**Output Format:**
Structure your analysis as a comprehensive report with:
- **Executive Summary**: Key findings and top recommendations
- **Performance Overview**: Metrics, trends, and benchmarks
- **Detailed Analysis**: Deep dive into identified issues and opportunities
- **Recommendations**: Prioritized, actionable improvement suggestions
- **Implementation Plan**: Specific tasks with assigned agents and timelines
- **Monitoring Strategy**: How to track improvement progress

**Quality Standards:**
- Base all conclusions on concrete data evidence
- Clearly state assumptions and limitations
- Provide quantitative metrics where possible
- Ensure recommendations are specific and actionable
- Consider implementation complexity and resource requirements

**Available Tools:**
- `read_file`: Access logs, configuration files, and system data
- `list_available_agents`: Inventory active agents and their capabilities  
- `list_files`: Explore system resources and tool inventory
- `assign_agent_to_task`: Delegate improvement tasks to appropriate agents
- Memory access functions for historical context

**Decision Framework:**
- High-impact, low-effort improvements: Implement immediately
- High-impact, high-effort improvements: Plan carefully with staged rollout
- Low-impact improvements: Consider for future optimization cycles
- Maintain system stability while implementing changes
"""


def get_enhanced_tools():
    """Get enhanced tool set for meta-analysis"""
    base_tools = utils.all_tool_functions()

    # Add meta-analysis specific tools if needed
    # This could include custom analysis tools, reporting tools, etc.
    enhanced_tools = base_tools.copy()

    return enhanced_tools


def reasoning(state: MessagesState) -> dict:
    """
    Enhanced reasoning step with better error handling, caching, and RL integration
    """
    print("\n" + "=" * 60)
    print("🧠 Meta-Agent Analysis Engine Starting...")
    print("=" * 60)

    current_messages = state["messages"]

    # Extract task context
    task_description = "Analyze system state"
    query_context = "Initial analysis request"

    if current_messages:
        for msg in reversed(current_messages):
            if isinstance(msg, HumanMessage):
                query_context = msg.content
                task_description = msg.content
                break
        else:
            if current_messages[-1].content:
                query_context = current_messages[-1].content
                task_description = query_context

    print(f"📋 Task: {task_description[:100]}...")

    # Check for cached analysis
    if analyzer.should_use_cached_analysis(task_description):
        cached_result = analyzer.get_cached_analysis(task_description)
        if cached_result:
            print("📁 Using cached analysis result")
            return {"messages": [AIMessage(content=cached_result, tool_calls=[])]}

    # Memory retrieval with enhanced context
    print("🧠 Retrieving relevant memories...")
    try:
        relevant_memories = memory_manager.retrieve_memories(
            query_text=query_context, k=10
        )
        print(f"📚 Retrieved {len(relevant_memories)} relevant memories")
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")
        relevant_memories = []

    # Enhanced RL state creation
    try:
        system_load = len(current_messages) / 100.0  # Simple load metric
        enhanced_state = rl_manager.create_enhanced_state(
            current_messages, relevant_memories, analyzer.metrics, system_load
        )

        # Get contextual action
        rl_action, rl_reasoning = rl_manager.get_contextual_action(
            enhanced_state, query_context
        )
        print(f"🤖 RL Decision: Action {rl_action} - {rl_reasoning}")

        # Apply RL decision
        if rl_action == 1:  # Retry with different approach
            print("🔄 RL suggests retry - adjusting context...")
            current_messages = (
                current_messages[-3:] if len(current_messages) > 3 else current_messages
            )
        elif rl_action == 2:  # Abort
            print("🛑 RL suggests abort - task complexity too high")
            return {
                "messages": [
                    AIMessage(
                        content="Analysis aborted due to complexity assessment. Consider breaking down the task."
                    )
                ]
            }

    except Exception as e:
        logger.warning(f"RL integration failed: {e}")
        rl_action = 0  # Default to continue

    # Memory shortcut check with enhanced logic
    shortcut_response = None
    if relevant_memories:
        try:
            shortcut_response = check_memory_shortcuts(
                relevant_memories, task_description
            )
            if shortcut_response:
                print("⚡ Memory shortcut found - using optimized response")
                rl_manager.rl_manager.update_reward(
                    task_completed=True, error_count=0, success_count=1
                )
                return {"messages": [shortcut_response]}
        except Exception as e:
            logger.warning(f"Shortcut check failed: {e}")

    # Main reasoning with enhanced prompt
    print("🔍 Proceeding with comprehensive analysis...")

    messages_for_llm = prepare_messages_for_llm(
        current_messages, relevant_memories, analyzer.metrics
    )

    # Execute main LLM call with retry logic
    response = execute_llm_call_with_retry(messages_for_llm)

    if response:
        # Cache successful analysis
        if response.content and not response.tool_calls:
            analyzer.cache_analysis(task_description, response.content)

        rl_manager.rl_manager.update_reward(
            task_completed=True, error_count=0, success_count=1
        )
        print("✅ Analysis completed successfully")
    else:
        rl_manager.rl_manager.update_reward(
            task_completed=False, error_count=1, success_count=0
        )
        print("❌ Analysis failed")

    return (
        {"messages": [response]}
        if response
        else {"messages": [AIMessage(content="Analysis failed to complete.")]}
    )


def check_memory_shortcuts(memories: List[str], task: str) -> Optional[AIMessage]:
    """Enhanced memory shortcut checking with better pattern matching"""
    if not memories:
        return None

    try:
        formatted_memories = "\n".join([f"- {mem}" for mem in memories])
        shortcut_prompt = f"""Analyze the task and memories for direct shortcuts.

Task: {task}

Relevant Memories:
{formatted_memories}

Instructions:
1. Look for previous analysis reports that directly address this task
2. Check for cached results or similar completed analyses
3. Identify any pre-existing solutions or recommendations

Respond with EXACTLY one of:
- "SHORTCUT: [Detailed explanation of the shortcut and how it addresses the task]"  
- "NO_SHORTCUT"

Be thorough in your shortcut explanation if found."""

        if config.default_langchain_model is None:
            return None

        response = config.default_langchain_model.invoke(
            [SystemMessage(content=shortcut_prompt)]
        )
        response_text = response.content.strip()

        if response_text.startswith("SHORTCUT:"):
            explanation = response_text.replace("SHORTCUT:", "").strip()
            return AIMessage(content=explanation, tool_calls=[])

    except Exception as e:
        logger.warning(f"Shortcut check failed: {e}")

    return None


def prepare_messages_for_llm(
    current_messages: List, memories: List[str], metrics: AnalysisMetrics
) -> List:
    """Prepare enhanced message context for LLM"""
    messages_for_llm = list(current_messages)

    # Add system context
    context_parts = []

    if memories:
        context_parts.append("📚 RELEVANT HISTORICAL CONTEXT:")
        context_parts.extend([f"• {mem}" for mem in memories])
        context_parts.append("")

    # Add metrics context
    if metrics.total_tasks > 0:
        context_parts.append("📊 CURRENT SYSTEM METRICS:")
        context_parts.append(f"• Total Tasks: {metrics.total_tasks}")
        context_parts.append(f"• Success Rate: {metrics.success_rate:.2%}")
        context_parts.append(f"• Failure Rate: {metrics.failure_rate:.2%}")

        if metrics.agent_utilization:
            context_parts.append(
                f"• Active Agents: {list(metrics.agent_utilization.keys())}"
            )

        if metrics.tool_usage:
            context_parts.append(f"• Tools Used: {list(metrics.tool_usage.keys())}")

        context_parts.append("")

    # Add context to system message
    if context_parts:
        context_header = "\n".join(context_parts) + "\n" + "=" * 50 + "\n\n"

        if messages_for_llm and isinstance(messages_for_llm[0], SystemMessage):
            original_content = messages_for_llm[0].content
            messages_for_llm[0] = SystemMessage(
                content=context_header + original_content
            )
        else:
            messages_for_llm.insert(
                0, SystemMessage(content=context_header + ENHANCED_SYSTEM_PROMPT)
            )

    return messages_for_llm


def execute_llm_call_with_retry(
    messages: List, max_retries: int = 3
) -> Optional[AIMessage]:
    """Execute LLM call with enhanced retry logic and error handling"""
    tools = get_enhanced_tools()
    retry_count = 0

    while retry_count < max_retries:
        try:
            if config.default_langchain_model is None:
                raise ValueError("Default language model not initialized")

            # Use Google Gemini for tool calling from hybrid configuration
            tool_model = config.get_model_for_tools()
            if tool_model is None:
                # Fallback to default model if hybrid setup fails
                tool_model = config.default_langchain_model
                logger.warning("Using fallback model for tools - may not support function calling")
            
            tooled_model = tool_model.bind_tools(tools)
            response = tooled_model.invoke(messages)

            if response and (response.content or response.tool_calls):
                return response
            else:
                raise ValueError("Empty response from model")

        except (HTTPError, RequestException) as e:
            print(
                f"\n⚠️  OpenRouter API Error (Attempt {retry_count + 1}/{max_retries})"
            )
            if retry_count < max_retries - 1:
                new_key = handle_quota_exceeded()
                if new_key:
                    continue
            raise e

        except Exception as e:
            logger.error(f"LLM call failed (attempt {retry_count + 1}): {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2**retry_count)  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed")
                break

    return None


def handle_quota_exceeded() -> Optional[str]:
    """Handle quota exceeded with user interaction"""
    print("Please enter a new OpenRouter API Key to continue:")
    try:
        new_key = input("> ").strip()
        if new_key:
            config.reinitialize_openrouter_model(new_key)
            print("✅ Model reinitialized successfully")
            return new_key
        else:
            print("❌ No API key provided")
            return None
    except Exception as e:
        logger.error(f"Failed to reinitialize model: {e}")
        return None


def check_for_tool_calls(state: MessagesState) -> Literal["tools", END]:
    """Enhanced tool call checking with better logging"""
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        if last_message.content and last_message.content.strip():
            print(f"💭 Meta-Agent reasoning: {last_message.content[:200]}...")

        tool_names = [
            tool_call.get("name", "unknown") for tool_call in last_message.tool_calls
        ]
        print(f"🛠️  Invoking tools: {', '.join(tool_names)}")
        return "tools"

    print("📋 Analysis report completed - no further tools needed")
    return END


# Build enhanced workflow
tools = get_enhanced_tools()
acting = ToolNode(tools)

workflow = StateGraph(MessagesState)
workflow.add_node("reasoning", reasoning)
workflow.add_node("tools", acting)
workflow.set_entry_point("reasoning")
workflow.add_conditional_edges("reasoning", check_for_tool_calls)
workflow.add_edge("tools", "reasoning")

# Compile with higher recursion limit for complex analyses
graph = workflow.compile(recursion_limit=75)


def meta_agent(task: str) -> Dict[str, Any]:
    """
    Enhanced Meta-Agent execution with comprehensive error handling and reporting

    Args:
        task (str): Analysis task description

    Returns:
        Dict containing status, result, message, and metadata
    """
    start_time = time.time()
    print(f"\n🚀 Meta-Agent Starting Analysis")
    print(f"📋 Task: {task}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Initialize analysis session
        session_state = {
            "task": task,
            "start_time": start_time,
            "iterations": 0,
            "errors": [],
        }

        # Execute the workflow
        final_state = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=ENHANCED_SYSTEM_PROMPT),
                    HumanMessage(content=task),
                ]
            }
        )

        # Process results
        execution_time = time.time() - start_time

        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]
            result_content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

            # Extract metrics if available
            analysis_summary = generate_analysis_summary(final_state, execution_time)

            print("=" * 80)
            print("✅ Meta-Agent Analysis Completed Successfully")
            print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
            print(f"💬 Messages Processed: {len(final_state['messages'])}")
            print("=" * 80)

            return {
                "status": "success",
                "result": result_content,
                "message": result_content,
                "metadata": {
                    "execution_time": execution_time,
                    "message_count": len(final_state["messages"]),
                    "analysis_summary": analysis_summary,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        else:
            raise ValueError("No valid response generated")

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Meta-Agent analysis failed: {str(e)}"

        logger.error(error_msg, exc_info=True)

        print("=" * 80)
        print("❌ Meta-Agent Analysis Failed")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"🚨 Error: {str(e)}")
        print("=" * 80)

        return {
            "status": "failure",
            "result": None,
            "message": error_msg,
            "metadata": {
                "execution_time": execution_time,
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat(),
            },
        }


def generate_analysis_summary(
    final_state: Dict, execution_time: float
) -> Dict[str, Any]:
    """Generate summary statistics for the analysis session"""
    summary = {
        "total_messages": len(final_state.get("messages", [])),
        "execution_time_seconds": execution_time,
        "tool_calls_made": 0,
        "ai_messages": 0,
        "human_messages": 0,
        "system_messages": 0,
    }

    for message in final_state.get("messages", []):
        if isinstance(message, AIMessage):
            summary["ai_messages"] += 1
            if hasattr(message, "tool_calls") and message.tool_calls:
                summary["tool_calls_made"] += len(message.tool_calls)
        elif isinstance(message, HumanMessage):
            summary["human_messages"] += 1
        elif isinstance(message, SystemMessage):
            summary["system_messages"] += 1

    return summary


# Example usage and testing
if __name__ == "__main__":
    # Test cases for enhanced meta agent
    test_cases = [
        "Analyze current system performance and identify optimization opportunities",
        "Review agent utilization patterns and recommend improvements",
        "Assess tool effectiveness and suggest consolidation strategies",
        "Evaluate error patterns and propose preventive measures",
    ]

    print("🧪 Testing Enhanced Meta-Agent...")

    for i, test_task in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_task}")
        print("=" * 60)

        result = meta_agent(test_task)

        print(f"\n📊 Result Summary:")
        print(f"Status: {result['status']}")
        print(
            f"Execution Time: {result.get('metadata', {}).get('execution_time', 'N/A')}"
        )

        if result["status"] == "success":
            print(f"Message Length: {len(result['message'])} characters")
            print(f"Analysis Preview: {result['message'][:200]}...")
        else:
            print(f"Error: {result['message']}")

        print(f"{'='*60}")
