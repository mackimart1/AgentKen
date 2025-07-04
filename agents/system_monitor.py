from typing import Dict, Any
import logging
import time
from tools.scratchpad import scratchpad
from tools.run_shell_command import run_shell_command
from tools.assign_agent_to_task import assign_agent_to_task

logger = logging.getLogger(__name__)


def system_monitor(task: str) -> Dict[str, Any]:
    """
    Monitors system metrics (CPU, memory, disk usage, agent activity), logs anomalies,
    and alerts other agents if intervention is needed. Also provides system status reports.

    Args:
        task (str): The task to perform (e.g., 'monitor', 'report', 'alert').

    Returns:
        Dict[str, Any]: Result dictionary with status, result, and message.
    """
    try:
        if task == "monitor":
            # Gather system metrics
            cpu_result = run_shell_command(
                "top -bn1 | grep \"Cpu(s)\" | awk '{print $2 + $4}'"
            )
            mem_result = run_shell_command("free -m | awk 'NR==2{print $3/$2 * 100.0}'")
            disk_result = run_shell_command(
                "df -h | awk '$NF==\"/\"{print $5}' | sed 's/%//g'"
            )

            # Log metrics to scratchpad
            timestamp = int(time.time())
            scratchpad(
                action="write",
                key=f"system_metrics_{timestamp}",
                value=f"CPU: {cpu_result['stdout']}%, Memory: {mem_result['stdout']}%, Disk: {disk_result['stdout']}%",
            )

            # Check for anomalies
            cpu_usage = float(cpu_result["stdout"].strip())
            mem_usage = float(mem_result["stdout"].strip())
            disk_usage = float(disk_result["stdout"].strip())

            if cpu_usage > 90 or mem_usage > 90 or disk_usage > 90:
                alert_message = f"System anomaly detected: CPU={cpu_usage}%, Memory={mem_usage}%, Disk={disk_usage}%"
                assign_agent_to_task(
                    agent_name="hermes", task=f"Alert: {alert_message}"
                )
                return {
                    "status": "success",
                    "result": alert_message,
                    "message": "System metrics logged and anomaly alert sent.",
                }

            return {
                "status": "success",
                "result": {"CPU": cpu_usage, "Memory": mem_usage, "Disk": disk_usage},
                "message": "System metrics logged successfully.",
            }

        elif task == "report":
            # Retrieve latest metrics from scratchpad
            keys = scratchpad(action="list_keys").get("result", [])
            latest_metrics = []
            for key in keys:
                if key.startswith("system_metrics_"):
                    metrics = scratchpad(action="read", key=key).get("result")
                    latest_metrics.append(metrics)

            return {
                "status": "success",
                "result": latest_metrics,
                "message": "Latest system metrics retrieved.",
            }

        else:
            return {
                "status": "failure",
                "result": None,
                "message": f"Unknown task: {task}. Valid tasks are 'monitor' and 'report'.",
            }

    except Exception as e:
        logger.error(f"Error in system_monitor: {e}")
        return {
            "status": "failure",
            "result": None,
            "message": f"Failed to complete task: {str(e)}",
        }
