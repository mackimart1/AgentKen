"""
AI Communication Module for Inferra V Enhanced System
Provides intelligent conversation capabilities using OpenRouter API with DeepSeek model.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

from config import config


@dataclass
class ConversationContext:
    """Context for maintaining conversation state"""

    session_id: str
    messages: List[Dict[str, str]]
    system_state: Dict[str, Any]
    last_interaction: float


class AIAssistant:
    """AI Assistant for intelligent communication with users"""

    def __init__(self):
        self.model_config = config.model
        self.conversations: Dict[str, ConversationContext] = {}
        self.system_prompt = self._create_system_prompt()

        # Override API key if needed (temporary fix)
        if self.model_config.api_key == "your_openrouter_api_key_here":
            self.model_config.api_key = "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"

        # Validate configuration
        if not self.model_config.api_key:
            raise ValueError("OpenRouter API key not configured")

        # Debug configuration
        logging.info(
            f"AI Assistant initialized with model: {self.model_config.model_name}"
        )
        logging.info(f"API Key configured: {self.model_config.api_key[:20]}...")
        logging.info(f"Base URL: {self.model_config.base_url}")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for the AI assistant"""
        return """You are Inferra V, an advanced AI orchestration assistant. You help users manage and interact with a sophisticated multi-agent system.

Your capabilities include:
- Creating and managing execution plans
- Monitoring system health and performance
- Coordinating multiple AI agents
- Optimizing workflows and resource allocation
- Providing real-time system insights

You should be:
- Helpful and informative
- Professional but friendly
- Concise but thorough when needed
- Proactive in suggesting optimizations
- Clear about system capabilities and limitations

Current system context:
- Multi-agent orchestration platform
- Real-time monitoring and optimization
- Task planning and execution
- Resource management
- Performance analytics

Respond naturally and helpfully to user queries about the system, its status, capabilities, and how to use it effectively."""

    def chat(
        self,
        message: str,
        session_id: str = "default",
        system_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a chat message and return an AI-generated response

        Args:
            message: User's message
            session_id: Conversation session identifier
            system_state: Current system state for context

        Returns:
            AI-generated response
        """
        try:
            # Get or create conversation context
            context = self._get_conversation_context(session_id, system_state)

            # Add user message to conversation
            context.messages.append({"role": "user", "content": message})

            # Generate response
            response = self._generate_response(context)

            # Add assistant response to conversation
            context.messages.append({"role": "assistant", "content": response})

            # Update conversation context
            context.last_interaction = time.time()
            context.system_state = system_state or {}

            # Keep conversation history manageable
            self._trim_conversation_history(context)

            return response

        except Exception as e:
            logging.error(f"AI chat error: {e}")
            return self._get_fallback_response(message)

    def _get_conversation_context(
        self, session_id: str, system_state: Optional[Dict[str, Any]]
    ) -> ConversationContext:
        """Get or create conversation context"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationContext(
                session_id=session_id,
                messages=[{"role": "system", "content": self.system_prompt}],
                system_state=system_state or {},
                last_interaction=time.time(),
            )

        return self.conversations[session_id]

    def _generate_response(self, context: ConversationContext) -> str:
        """Generate AI response using OpenRouter API"""

        # Prepare messages with system context
        messages = context.messages.copy()

        # Add current system state as context if available
        if context.system_state:
            system_context = self._format_system_context(context.system_state)
            messages.append(
                {
                    "role": "system",
                    "content": f"Current system status: {system_context}",
                }
            )

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.model_config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://inferra-v.local",
            "X-Title": "Inferra V Enhanced System",
        }

        payload = {
            "model": self.model_config.model_name,
            "messages": messages,
            "temperature": self.model_config.temperature,
            "max_tokens": 500,
            "stream": False,
        }

        # Make API request
        try:
            response = requests.post(
                f"{self.model_config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            # Check for specific error codes
            if response.status_code == 401:
                raise Exception(
                    f"API Authentication failed. Please check your OpenRouter API key. Status: {response.status_code}"
                )
            elif response.status_code == 429:
                raise Exception("API rate limit exceeded. Please try again later.")
            elif response.status_code >= 400:
                error_detail = response.text if response.text else "Unknown error"
                raise Exception(
                    f"API request failed with status {response.status_code}: {error_detail}"
                )

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error connecting to OpenRouter API: {e}")

        # Extract response
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise Exception("Invalid API response format")

    def _format_system_context(self, system_state: Dict[str, Any]) -> str:
        """Format system state for AI context"""
        context_parts = []

        if "health_score" in system_state:
            context_parts.append(f"Health: {system_state['health_score']:.1f}/100")

        if "active_plans" in system_state:
            context_parts.append(f"Active plans: {system_state['active_plans']}")

        if "total_plans" in system_state:
            context_parts.append(f"Total plans: {system_state['total_plans']}")

        if "agents" in system_state:
            context_parts.append(f"Agents: {system_state['agents']}")

        if "active_alerts" in system_state:
            context_parts.append(f"Alerts: {system_state['active_alerts']}")

        return ", ".join(context_parts) if context_parts else "No system data available"

    def _trim_conversation_history(
        self, context: ConversationContext, max_messages: int = 20
    ):
        """Keep conversation history manageable"""
        if len(context.messages) > max_messages:
            # Keep system prompt and recent messages
            system_msg = context.messages[0]  # System prompt
            recent_messages = context.messages[-(max_messages - 1) :]
            context.messages = [system_msg] + recent_messages

    def _get_fallback_response(self, message: str) -> str:
        """Provide fallback response when AI is unavailable"""
        message_lower = message.lower()

        # Simple rule-based fallback responses
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Inferra V, your AI orchestration assistant. How can I help you today?"

        elif any(word in message_lower for word in ["status", "health", "how"]):
            return "I can help you check system status. Try using the 'status' command to see detailed system health information."

        elif any(word in message_lower for word in ["help", "what", "can"]):
            return "I'm here to help you manage the Inferra V system! I can assist with creating plans, monitoring execution, optimizing performance, and answering questions about the system. Try commands like 'status', 'create_plan', or ask me specific questions!"

        elif any(word in message_lower for word in ["plan", "task", "create"]):
            return "I can help you create and manage execution plans! Use the 'create_plan' command to start building a new plan, or ask me about specific planning strategies."

        elif any(
            word in message_lower for word in ["optimize", "improve", "performance"]
        ):
            return "I can provide optimization recommendations! Use the 'optimize' command to get detailed suggestions for improving system performance."

        else:
            return "I understand you want to interact with the system. I'm currently running in fallback mode. Try specific commands like 'status', 'help', or 'create_plan', or ask me questions about the Inferra V system!"

    def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of conversation"""
        if session_id not in self.conversations:
            return None

        context = self.conversations[session_id]

        return {
            "session_id": session_id,
            "message_count": len(context.messages) - 1,  # Exclude system prompt
            "last_interaction": context.last_interaction,
            "duration": time.time() - context.last_interaction,
        }

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False


class SmartCommandProcessor:
    """Process natural language commands and convert them to system actions"""

    def __init__(self, ai_assistant: AIAssistant):
        self.ai_assistant = ai_assistant
        self.command_patterns = {
            "status": ["status", "health", "how is", "system state"],
            "create_plan": ["create plan", "new plan", "make plan", "build plan"],
            "list_plans": ["list plans", "show plans", "what plans", "plans available"],
            "execute_plan": ["execute", "run plan", "start plan"],
            "optimize": ["optimize", "improve", "recommendations", "suggestions"],
            "help": ["help", "what can", "commands", "how to"],
        }

    def process_natural_language(
        self, message: str, system_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process natural language input and determine appropriate action

        Returns:
            Dict with 'type' (command/chat) and 'action' or 'response'
        """
        message_lower = message.lower().strip()

        # Check for direct commands
        for command, patterns in self.command_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                return {
                    "type": "command",
                    "action": command,
                    "original_message": message,
                }

        # If no command pattern matched, treat as chat
        response = self.ai_assistant.chat(message, system_state=system_state)

        return {"type": "chat", "response": response, "original_message": message}


# Global AI assistant instance
_ai_assistant = None


def get_ai_assistant() -> AIAssistant:
    """Get global AI assistant instance"""
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = AIAssistant()
    return _ai_assistant


def create_smart_processor() -> SmartCommandProcessor:
    """Create smart command processor"""
    return SmartCommandProcessor(get_ai_assistant())


# Test function
def test_ai_communication():
    """Test AI communication functionality"""
    try:
        assistant = AIAssistant()

        # Test basic chat
        response = assistant.chat("Hello, how are you?")
        print(f"AI Response: {response}")

        # Test with system context
        system_state = {"health_score": 96.2, "active_plans": 1, "agents": 2}

        response = assistant.chat(
            "What's the system status?", system_state=system_state
        )
        print(f"AI Response with context: {response}")

        return True

    except Exception as e:
        print(f"AI communication test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the AI communication
    logging.basicConfig(level=logging.INFO)
    test_ai_communication()
