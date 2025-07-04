#!/usr/bin/env python3
"""
Automated User Setup Script - Phase 2 Onboarding
This script helps system administrators create user accounts and assign roles.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.roles import RoleManager, User, PermissionError


class UserSetupManager:
    """Manages user account creation and role assignment."""

    def __init__(self):
        self.role_manager = RoleManager()
        self.users_file = "config/users.json"
        self.audit_file = "logs/user_setup_audit.json"

    def load_users_config(self) -> Dict:
        """Load user configuration from file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, "r") as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"❌ Error loading users config: {e}")
            return {}

    def save_users_config(self, config: Dict):
        """Save user configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            with open(self.users_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving users config: {e}")

    def log_audit_event(self, event: Dict):
        """Log audit events for user management."""
        try:
            os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)

            # Load existing audit log
            audit_log = []
            if os.path.exists(self.audit_file):
                with open(self.audit_file, "r") as f:
                    audit_log = json.load(f)

            # Add timestamp
            event["timestamp"] = datetime.now().isoformat()
            audit_log.append(event)

            # Save audit log
            with open(self.audit_file, "w") as f:
                json.dump(audit_log, f, indent=2)

        except Exception as e:
            print(f"⚠️ Warning: Could not log audit event: {e}")

    def create_user(
        self,
        username: str,
        role: str,
        email: str = "",
        full_name: str = "",
        github_username: str = "",
    ) -> bool:
        """Create a new user with specified role."""
        try:
            # Validate role
            valid_roles = ["tool_maker", "agent_smith", "admin"]
            if role not in valid_roles:
                print(f"❌ Invalid role: {role}. Valid roles: {valid_roles}")
                return False

            # Check if user already exists
            try:
                existing_user = self.role_manager.get_user(username)
                print(
                    f"⚠️ User '{username}' already exists with role '{existing_user.role}'"
                )
                return False
            except:
                pass  # User doesn't exist, continue with creation

            # Create user
            self.role_manager.create_user(username, role)

            # Get created user to verify
            user = self.role_manager.get_user(username)

            # Save user details to config
            config = self.load_users_config()
            config[username] = {
                "role": role,
                "email": email,
                "full_name": full_name,
                "github_username": github_username,
                "created_at": datetime.now().isoformat(),
                "status": "active",
            }
            self.save_users_config(config)

            # Log audit event
            self.log_audit_event(
                {
                    "action": "create_user",
                    "username": username,
                    "role": role,
                    "email": email,
                    "full_name": full_name,
                    "github_username": github_username,
                    "admin": "system_admin",
                }
            )

            print(f"✅ User '{username}' created successfully with role '{role}'")
            print(f"   Email: {email}")
            print(f"   Full Name: {full_name}")
            print(f"   GitHub: {github_username}")
            print(f"   Permissions: {user.get_permissions()}")

            return True

        except Exception as e:
            print(f"❌ Error creating user '{username}': {e}")
            return False

    def update_user_role(self, username: str, new_role: str) -> bool:
        """Update user role."""
        try:
            # Validate role
            valid_roles = ["tool_maker", "agent_smith", "admin"]
            if new_role not in valid_roles:
                print(f"❌ Invalid role: {new_role}. Valid roles: {valid_roles}")
                return False

            # Check if user exists
            try:
                user = self.role_manager.get_user(username)
                old_role = user.role
            except:
                print(f"❌ User '{username}' does not exist")
                return False

            # Update role
            self.role_manager.update_user_role(username, new_role)

            # Update config
            config = self.load_users_config()
            if username in config:
                config[username]["role"] = new_role
                config[username]["updated_at"] = datetime.now().isoformat()
                self.save_users_config(config)

            # Log audit event
            self.log_audit_event(
                {
                    "action": "update_role",
                    "username": username,
                    "old_role": old_role,
                    "new_role": new_role,
                    "admin": "system_admin",
                }
            )

            print(
                f"✅ User '{username}' role updated from '{old_role}' to '{new_role}'"
            )
            return True

        except Exception as e:
            print(f"❌ Error updating user '{username}': {e}")
            return False

    def delete_user(self, username: str) -> bool:
        """Delete a user account."""
        try:
            # Check if user exists
            try:
                user = self.role_manager.get_user(username)
                role = user.role
            except:
                print(f"❌ User '{username}' does not exist")
                return False

            # Delete user
            self.role_manager.delete_user(username)

            # Update config
            config = self.load_users_config()
            if username in config:
                config[username]["status"] = "deleted"
                config[username]["deleted_at"] = datetime.now().isoformat()
                self.save_users_config(config)

            # Log audit event
            self.log_audit_event(
                {
                    "action": "delete_user",
                    "username": username,
                    "role": role,
                    "admin": "system_admin",
                }
            )

            print(f"✅ User '{username}' deleted successfully")
            return True

        except Exception as e:
            print(f"❌ Error deleting user '{username}': {e}")
            return False

    def list_users(self, show_deleted: bool = False) -> None:
        """List all users and their details."""
        try:
            config = self.load_users_config()

            if not config:
                print("📋 No users found in configuration")
                return

            print("📋 User List:")
            print("-" * 80)
            print(
                f"{'Username':<15} {'Role':<12} {'Email':<25} {'Full Name':<20} {'Status':<10}"
            )
            print("-" * 80)

            for username, details in config.items():
                if not show_deleted and details.get("status") == "deleted":
                    continue

                role = details.get("role", "unknown")
                email = details.get("email", "")
                full_name = details.get("full_name", "")
                status = details.get("status", "active")

                print(
                    f"{username:<15} {role:<12} {email:<25} {full_name:<20} {status:<10}"
                )

            print("-" * 80)

        except Exception as e:
            print(f"❌ Error listing users: {e}")

    def get_user_details(self, username: str) -> None:
        """Get detailed information about a specific user."""
        try:
            # Get user from role manager
            try:
                user = self.role_manager.get_user(username)
                permissions = user.get_permissions()
            except:
                print(f"❌ User '{username}' not found in role manager")
                return

            # Get additional details from config
            config = self.load_users_config()
            details = config.get(username, {})

            print(f"👤 User Details: {username}")
            print("-" * 50)
            print(f"Role: {user.role}")
            print(f"Permissions: {permissions}")
            print(f"Email: {details.get('email', 'Not specified')}")
            print(f"Full Name: {details.get('full_name', 'Not specified')}")
            print(f"GitHub: {details.get('github_username', 'Not specified')}")
            print(f"Status: {details.get('status', 'active')}")
            print(f"Created: {details.get('created_at', 'Unknown')}")
            if "updated_at" in details:
                print(f"Updated: {details['updated_at']}")

        except Exception as e:
            print(f"❌ Error getting user details: {e}")

    def bulk_create_users(self, users_file: str) -> bool:
        """Create multiple users from a JSON file."""
        try:
            if not os.path.exists(users_file):
                print(f"❌ Users file not found: {users_file}")
                return False

            with open(users_file, "r") as f:
                users_data = json.load(f)

            if not isinstance(users_data, list):
                print("❌ Users file should contain a list of user objects")
                return False

            success_count = 0
            total_count = len(users_data)

            print(f"🚀 Creating {total_count} users...")

            for user_data in users_data:
                username = user_data.get("username")
                role = user_data.get("role")
                email = user_data.get("email", "")
                full_name = user_data.get("full_name", "")
                github_username = user_data.get("github_username", "")

                if not username or not role:
                    print(f"⚠️ Skipping user: missing username or role")
                    continue

                if self.create_user(username, role, email, full_name, github_username):
                    success_count += 1

            print(f"✅ Successfully created {success_count}/{total_count} users")
            return success_count == total_count

        except Exception as e:
            print(f"❌ Error in bulk user creation: {e}")
            return False

    def generate_user_report(self) -> None:
        """Generate a comprehensive user report."""
        try:
            config = self.load_users_config()

            if not config:
                print("📊 No users to report")
                return

            # Count by role
            role_counts = {}
            active_users = 0
            deleted_users = 0

            for username, details in config.items():
                role = details.get("role", "unknown")
                status = details.get("status", "active")

                role_counts[role] = role_counts.get(role, 0) + 1

                if status == "active":
                    active_users += 1
                else:
                    deleted_users += 1

            print("📊 User Report")
            print("=" * 50)
            print(f"Total Users: {len(config)}")
            print(f"Active Users: {active_users}")
            print(f"Deleted Users: {deleted_users}")
            print()
            print("Users by Role:")
            for role, count in role_counts.items():
                print(f"  {role}: {count}")
            print()
            print("Recent Activity:")

            # Show recent audit events
            if os.path.exists(self.audit_file):
                with open(self.audit_file, "r") as f:
                    audit_log = json.load(f)

                recent_events = audit_log[-5:]  # Last 5 events
                for event in recent_events:
                    action = event.get("action", "unknown")
                    username = event.get("username", "unknown")
                    timestamp = event.get("timestamp", "unknown")
                    print(f"  {timestamp[:19]}: {action} - {username}")

        except Exception as e:
            print(f"❌ Error generating report: {e}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="User Setup Manager for Permissioned Creation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a single user
  python setup_users.py create --username john_doe --role tool_maker --email john@company.com --full-name "John Doe"

  # Create multiple users from file
  python setup_users.py bulk-create --file users.json

  # List all users
  python setup_users.py list

  # Get user details
  python setup_users.py details --username john_doe

  # Update user role
  python setup_users.py update-role --username john_doe --role agent_smith

  # Delete user
  python setup_users.py delete --username john_doe

  # Generate report
  python setup_users.py report
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create user command
    create_parser = subparsers.add_parser("create", help="Create a single user")
    create_parser.add_argument("--username", required=True, help="Username")
    create_parser.add_argument(
        "--role",
        required=True,
        choices=["tool_maker", "agent_smith", "admin"],
        help="User role",
    )
    create_parser.add_argument("--email", default="", help="Email address")
    create_parser.add_argument("--full-name", default="", help="Full name")
    create_parser.add_argument("--github-username", default="", help="GitHub username")

    # Bulk create command
    bulk_parser = subparsers.add_parser(
        "bulk-create", help="Create multiple users from file"
    )
    bulk_parser.add_argument("--file", required=True, help="JSON file with user data")

    # List users command
    list_parser = subparsers.add_parser("list", help="List all users")
    list_parser.add_argument(
        "--show-deleted", action="store_true", help="Include deleted users"
    )

    # User details command
    details_parser = subparsers.add_parser("details", help="Get user details")
    details_parser.add_argument("--username", required=True, help="Username")

    # Update role command
    update_parser = subparsers.add_parser("update-role", help="Update user role")
    update_parser.add_argument("--username", required=True, help="Username")
    update_parser.add_argument(
        "--role",
        required=True,
        choices=["tool_maker", "agent_smith", "admin"],
        help="New role",
    )

    # Delete user command
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("--username", required=True, help="Username")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate user report")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize manager
    manager = UserSetupManager()

    # Execute command
    if args.command == "create":
        success = manager.create_user(
            args.username, args.role, args.email, args.full_name, args.github_username
        )
        sys.exit(0 if success else 1)

    elif args.command == "bulk-create":
        success = manager.bulk_create_users(args.file)
        sys.exit(0 if success else 1)

    elif args.command == "list":
        manager.list_users(args.show_deleted)

    elif args.command == "details":
        manager.get_user_details(args.username)

    elif args.command == "update-role":
        success = manager.update_user_role(args.username, args.role)
        sys.exit(0 if success else 1)

    elif args.command == "delete":
        success = manager.delete_user(args.username)
        sys.exit(0 if success else 1)

    elif args.command == "report":
        manager.generate_user_report()


if __name__ == "__main__":
    main()
