import json
from typing import List, Dict, Any, Optional


class ManifestManager:
    def __init__(self, api_client):
        self.api_client = api_client
        self.manifest_path = "agents_manifest.json"

    def _read_manifest(self) -> Dict[str, Any]:
        try:
            content = self.api_client.read_file(file_path=self.manifest_path)
            if content and "content" in content:
                return json.loads(content["content"])
            return {"agents": []}
        except FileNotFoundError:
            return {"agents": []}
        except json.JSONDecodeError:
            print(
                f"Error: Malformed JSON in {self.manifest_path}. Starting with empty manifest."
            )
            return {"agents": []}
        except Exception as e:
            print(f"An unexpected error occurred while reading manifest: {e}")
            return {"agents": []}

    def _write_manifest(self, manifest_data: Dict[str, Any]) -> bool:
        try:
            result = self.api_client.write_to_file(
                file=self.manifest_path,
                file_contents=json.dumps(manifest_data, indent=2),
            )
            return (
                "success" in result.get("status", "").lower()
                or "successfully" in result.get("message", "").lower()
            )
        except Exception as e:
            print(f"An error occurred while writing manifest: {e}")
            return False

    def add_agent(
        self, name: str, description: str, capabilities: List[str]
    ) -> Dict[str, Any]:
        manifest = self._read_manifest()
        agents = manifest.get("agents", [])

        if any(agent.get("name") == name for agent in agents):
            return {"status": "error", "message": f"Agent '{name}' already exists."}

        new_agent = {
            "name": name,
            "description": description,
            "capabilities": capabilities,
        }
        agents.append(new_agent)
        manifest["agents"] = agents

        if self._write_manifest(manifest):
            return {
                "status": "success",
                "message": f"Agent '{name}' added successfully.",
            }
        else:
            return {"status": "error", "message": f"Failed to add agent '{name}'."}

    def update_agent(
        self, name: str, new_description: Optional[str] = None, new_capabilities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        manifest = self._read_manifest()
        agents = manifest.get("agents", [])
        updated = False

        for agent in agents:
            if agent.get("name") == name:
                if new_description is not None:
                    agent["description"] = new_description
                if new_capabilities is not None:
                    agent["capabilities"] = new_capabilities
                updated = True
                break

        if not updated:
            return {"status": "error", "message": f"Agent '{name}' not found."}

        manifest["agents"] = agents
        if self._write_manifest(manifest):
            return {
                "status": "success",
                "message": f"Agent '{name}' updated successfully.",
            }
        else:
            return {"status": "error", "message": f"Failed to update agent '{name}'."}

    def remove_agent(self, name: str) -> Dict[str, Any]:
        manifest = self._read_manifest()
        agents = manifest.get("agents", [])
        original_len = len(agents)
        agents = [agent for agent in agents if agent.get("name") != name]
        manifest["agents"] = agents

        if len(agents) == original_len:
            return {"status": "error", "message": f"Agent '{name}' not found."}

        if self._write_manifest(manifest):
            return {
                "status": "success",
                "message": f"Agent '{name}' removed successfully.",
            }
        else:
            return {"status": "error", "message": f"Failed to remove agent '{name}'."}

    def get_agent(self, name: str) -> Dict[str, Any]:
        manifest = self._read_manifest()
        agents = manifest.get("agents", [])
        for agent in agents:
            if agent.get("name") == name:
                return {"status": "success", "agent": agent}
        return {"status": "error", "message": f"Agent '{name}' not found."}

    def list_agents(self) -> Dict[str, Any]:
        manifest = self._read_manifest()
        return {"status": "success", "agents": manifest.get("agents", [])}
