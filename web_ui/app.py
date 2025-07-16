#!/usr/bin/env python3
"""
Web UI for Permissioned Agent/Tool Creation System

A modern web interface for creating agents and tools with permission enforcement,
real-time validation, and a user-friendly experience.
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

from core.roles import role_manager, UserRole, Permission, log_audit_event
from create_entity import EntityCreator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-this")
app.config["SESSION_TYPE"] = "filesystem"

# Enable CORS for development
CORS(app)

# Initialize components
creator = EntityCreator()


class WebUIAuth:
    """Simple authentication for web UI"""

    def __init__(self):
        self.users_file = "web_ui/users.json"
        self.users = self._load_users()

    def _load_users(self) -> Dict[str, Dict[str, Any]]:
        """Load users from file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
        return {}

    def _save_users(self):
        """Save users to file"""
        try:
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")

    def register_user(self, username: str, password: str, role: str) -> bool:
        """Register a new user"""
        if username in self.users:
            return False

        # Create user in permissioned system
        try:
            user_role = UserRole(role)
            role_manager.set_user_role(username, user_role, created_by="web_ui")
        except ValueError:
            return False

        # Store web UI user
        self.users[username] = {
            "password_hash": generate_password_hash(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
        }

        self._save_users()
        return True

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user"""
        if username not in self.users:
            return False

        user = self.users[username]
        if check_password_hash(user["password_hash"], password):
            user["last_login"] = datetime.now().isoformat()
            self._save_users()
            return True

        return False

    def get_user_role(self, username: str) -> Optional[str]:
        """Get user role"""
        if username in self.users:
            return self.users[username]["role"]
        return None


auth = WebUIAuth()


@app.route("/")
def index():
    """Main dashboard page"""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    user_role = auth.get_user_role(user_id)

    # Get system statistics
    entities = creator.list_entities("all")
    agents = (
        entities.get("result", {}).get("agents", [])
        if entities.get("status") == "success"
        else []
    )
    tools = (
        entities.get("result", {}).get("tools", [])
        if entities.get("status") == "success"
        else []
    )

    # Get user permissions
    permissions = role_manager.get_user_permissions(user_id)
    permission_names = [p.value for p in permissions]

    return render_template(
        "dashboard.html",
        user_id=user_id,
        user_role=user_role,
        agents=agents,
        tools=tools,
        permissions=permission_names,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if auth.authenticate(username, password):
            session["user_id"] = username
            log_audit_event(username, "login", "web_ui", {"method": "web_ui"})
            return jsonify({"success": True, "redirect": url_for("index")})
        else:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logout user"""
    if "user_id" in session:
        log_audit_event(session["user_id"], "logout", "web_ui", {"method": "web_ui"})
        session.pop("user_id", None)

    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register new user"""
    if request.method == "POST":
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")
        role = data.get("role", "viewer")

        if auth.register_user(username, password, role):
            return jsonify({"success": True, "message": "User registered successfully"})
        else:
            return jsonify({"success": False, "error": "Username already exists"}), 400

    return render_template("register.html")


@app.route("/api/create-agent", methods=["POST"])
def create_agent_api():
    """API endpoint for creating agents"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]

    # Check permissions
    if not role_manager.check_permission(user_id, Permission.CREATE_AGENT):
        return jsonify({"error": "Insufficient permissions"}), 403

    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "description", "capabilities"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create agent
        result = creator.create_agent(
            user_id=user_id,
            agent_name=data["name"],
            description=data["description"],
            capabilities=data["capabilities"],
            author=data.get("author", user_id),
        )

        if result["status"] == "success":
            log_audit_event(
                user_id,
                "create_agent",
                data["name"],
                {
                    "description": data["description"],
                    "capabilities": data["capabilities"],
                },
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/create-tool", methods=["POST"])
def create_tool_api():
    """API endpoint for creating tools"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]

    # Check permissions
    if not role_manager.check_permission(user_id, Permission.CREATE_TOOL):
        return jsonify({"error": "Insufficient permissions"}), 403

    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "description", "parameters", "return_type"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Create tool
        result = creator.create_tool(
            user_id=user_id,
            tool_name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            return_type=data["return_type"],
            author=data.get("author", user_id),
        )

        if result["status"] == "success":
            log_audit_event(
                user_id,
                "create_tool",
                data["name"],
                {
                    "description": data["description"],
                    "parameters": data["parameters"],
                    "return_type": data["return_type"],
                },
            )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error creating tool: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/entities")
def list_entities_api():
    """API endpoint for listing entities"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    entity_type = request.args.get("type", "all")

    # Check permissions
    if entity_type in ["agents", "all"] and not role_manager.check_permission(
        user_id, Permission.VIEW_AGENTS
    ):
        return jsonify({"error": "Insufficient permissions"}), 403

    if entity_type in ["tools", "all"] and not role_manager.check_permission(
        user_id, Permission.VIEW_TOOLS
    ):
        return jsonify({"error": "Insufficient permissions"}), 403

    result = creator.list_entities(entity_type)
    return jsonify(result)


@app.route("/api/user-info")
def user_info_api():
    """API endpoint for user information"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]
    user_role = auth.get_user_role(user_id)
    permissions = role_manager.get_user_permissions(user_id)

    return jsonify(
        {
            "user_id": user_id,
            "role": user_role,
            "permissions": [p.value for p in permissions],
        }
    )


@app.route("/api/audit-logs")
def audit_logs_api():
    """API endpoint for audit logs"""
    if "user_id" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    user_id = session["user_id"]

    # Check permissions
    if not role_manager.check_permission(user_id, Permission.VIEW_AUDIT_LOGS):
        return jsonify({"error": "Insufficient permissions"}), 403

    try:
        # Read audit log file
        audit_file = "audit.log"
        logs = []

        if os.path.exists(audit_file):
            with open(audit_file, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue

        # Return recent logs (last 100)
        return jsonify({"logs": logs[-100:]})

    except Exception as e:
        logger.error(f"Error reading audit logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/create-agent")
def create_agent_page():
    """Agent creation page"""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not role_manager.check_permission(user_id, Permission.CREATE_AGENT):
        return render_template(
            "error.html",
            error="Insufficient permissions",
            message="You don't have permission to create agents.",
        )

    return render_template("create_agent.html", user_id=user_id)


@app.route("/create-tool")
def create_tool_page():
    """Tool creation page"""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not role_manager.check_permission(user_id, Permission.CREATE_TOOL):
        return render_template(
            "error.html",
            error="Insufficient permissions",
            message="You don't have permission to create tools.",
        )

    return render_template("create_tool.html", user_id=user_id)


@app.route("/entities")
def entities_page():
    """Entities listing page"""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not (
        role_manager.check_permission(user_id, Permission.VIEW_AGENTS)
        or role_manager.check_permission(user_id, Permission.VIEW_TOOLS)
    ):
        return render_template(
            "error.html",
            error="Insufficient permissions",
            message="You don't have permission to view entities.",
        )

    return render_template("entities.html", user_id=user_id)


@app.route("/audit-logs")
def audit_logs_page():
    """Audit logs page"""
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    if not role_manager.check_permission(user_id, Permission.VIEW_AUDIT_LOGS):
        return render_template(
            "error.html",
            error="Insufficient permissions",
            message="You don't have permission to view audit logs.",
        )

    return render_template("audit_logs.html", user_id=user_id)


@app.errorhandler(404)
def not_found(error):
    return (
        render_template(
            "error.html",
            error="Page Not Found",
            message="The page you're looking for doesn't exist.",
        ),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        render_template(
            "error.html",
            error="Internal Server Error",
            message="Something went wrong on our end.",
        ),
        500,
    )


if __name__ == "__main__":
    # Create default admin user if needed
    if not auth.users:
        auth.register_user("admin", "admin123", "admin")
        logger.info("Created default admin user: admin/admin123")

    app.run(debug=True, host="0.0.0.0", port=5000)
