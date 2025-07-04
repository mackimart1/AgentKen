#!/usr/bin/env python3
"""
Test Script for Permissioned Agent/Tool Creation System

This script demonstrates the complete workflow for creating agents and tools
with permission enforcement, template application, and audit logging.
"""
import json
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.roles import role_manager, UserRole, Permission
from create_entity import EntityCreator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_test_users():
    """Set up test users with different roles"""
    logger.info("Setting up test users...")

    # Create admin user
    role_manager.set_user_role("admin_user", UserRole.ADMIN)

    # Create agent smith user
    role_manager.set_user_role("agent_smith_user", UserRole.AGENT_SMITH)

    # Create tool maker user
    role_manager.set_user_role("tool_maker_user", UserRole.TOOL_MAKER)

    # Create viewer user
    role_manager.set_user_role("viewer_user", UserRole.VIEWER)

    logger.info("Test users created successfully")


def test_permission_checks():
    """Test permission checking functionality"""
    logger.info("Testing permission checks...")

    # Test admin permissions
    assert role_manager.check_permission("admin_user", Permission.CREATE_AGENT)
    assert role_manager.check_permission("admin_user", Permission.CREATE_TOOL)
    assert role_manager.check_permission("admin_user", Permission.MANAGE_USERS)

    # Test agent smith permissions
    assert role_manager.check_permission("agent_smith_user", Permission.CREATE_AGENT)
    assert not role_manager.check_permission("agent_smith_user", Permission.CREATE_TOOL)
    assert not role_manager.check_permission(
        "agent_smith_user", Permission.MANAGE_USERS
    )

    # Test tool maker permissions
    assert not role_manager.check_permission("tool_maker_user", Permission.CREATE_AGENT)
    assert role_manager.check_permission("tool_maker_user", Permission.CREATE_TOOL)
    assert not role_manager.check_permission("tool_maker_user", Permission.MANAGE_USERS)

    # Test viewer permissions
    assert not role_manager.check_permission("viewer_user", Permission.CREATE_AGENT)
    assert not role_manager.check_permission("viewer_user", Permission.CREATE_TOOL)
    assert role_manager.check_permission("viewer_user", Permission.VIEW_AGENTS)

    logger.info("Permission checks passed!")


def test_agent_creation():
    """Test agent creation with permission enforcement"""
    logger.info("Testing agent creation...")

    creator = EntityCreator()

    # Test successful agent creation by agent smith
    result = creator.create_agent(
        user_id="agent_smith_user",
        agent_name="Test Data Processor",
        description="Processes and analyzes test data",
        capabilities=["data_processing", "analysis", "reporting"],
        author="Test Agent Smith",
    )

    print(f"Agent creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("Agent creation successful!")
    else:
        logger.error(f"Agent creation failed: {result['message']}")

    # Test failed agent creation by tool maker (should fail)
    result = creator.create_agent(
        user_id="tool_maker_user",
        agent_name="Unauthorized Agent",
        description="This should fail",
        capabilities=["test"],
        author="Test Tool Maker",
    )

    print(f"Unauthorized agent creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "failure":
        logger.info("Unauthorized agent creation correctly blocked!")
    else:
        logger.error("Unauthorized agent creation should have failed!")


def test_tool_creation():
    """Test tool creation with permission enforcement"""
    logger.info("Testing tool creation...")

    creator = EntityCreator()

    # Test successful tool creation by tool maker
    result = creator.create_tool(
        user_id="tool_maker_user",
        tool_name="Test Data Validator",
        description="Validates test data format and content",
        parameters={"data": "str", "format": "str", "strict": "bool"},
        return_type="bool",
        author="Test Tool Maker",
    )

    print(f"Tool creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("Tool creation successful!")
    else:
        logger.error(f"Tool creation failed: {result['message']}")

    # Test failed tool creation by agent smith (should fail)
    result = creator.create_tool(
        user_id="agent_smith_user",
        tool_name="Unauthorized Tool",
        description="This should fail",
        parameters={"test": "str"},
        return_type="str",
        author="Test Agent Smith",
    )

    print(f"Unauthorized tool creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "failure":
        logger.info("Unauthorized tool creation correctly blocked!")
    else:
        logger.error("Unauthorized tool creation should have failed!")


def test_admin_creation():
    """Test that admin can create both agents and tools"""
    logger.info("Testing admin creation capabilities...")

    creator = EntityCreator()

    # Test admin creating an agent
    result = creator.create_agent(
        user_id="admin_user",
        agent_name="Admin Test Agent",
        description="Agent created by admin",
        capabilities=["admin_test"],
        author="Admin User",
    )

    print(f"Admin agent creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("Admin agent creation successful!")
    else:
        logger.error(f"Admin agent creation failed: {result['message']}")

    # Test admin creating a tool
    result = creator.create_tool(
        user_id="admin_user",
        tool_name="Admin Test Tool",
        description="Tool created by admin",
        parameters={"admin_param": "str"},
        return_type="str",
        author="Admin User",
    )

    print(f"Admin tool creation result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("Admin tool creation successful!")
    else:
        logger.error(f"Admin tool creation failed: {result['message']}")


def test_listing_entities():
    """Test entity listing functionality"""
    logger.info("Testing entity listing...")

    creator = EntityCreator()

    # List all entities
    result = creator.list_entities("all")

    print(f"Entity listing result: {json.dumps(result, indent=2)}")

    if result["status"] == "success":
        logger.info("Entity listing successful!")
        agents = result["result"].get("agents", [])
        tools = result["result"].get("tools", [])
        logger.info(f"Found {len(agents)} agents and {len(tools)} tools")
    else:
        logger.error(f"Entity listing failed: {result['message']}")


def test_user_management():
    """Test user management functionality"""
    logger.info("Testing user management...")

    # List all users
    users = role_manager.list_users()
    print(f"Current users: {json.dumps(users, indent=2)}")

    # Test getting user permissions
    admin_permissions = role_manager.get_user_permissions("admin_user")
    agent_smith_permissions = role_manager.get_user_permissions("agent_smith_user")
    tool_maker_permissions = role_manager.get_user_permissions("tool_maker_user")

    print(f"Admin permissions: {[p.value for p in admin_permissions]}")
    print(f"Agent Smith permissions: {[p.value for p in agent_smith_permissions]}")
    print(f"Tool Maker permissions: {[p.value for p in tool_maker_permissions]}")

    logger.info("User management tests completed!")


def main():
    """Main test function"""
    logger.info("Starting Permissioned Creation System Tests")
    logger.info("=" * 50)

    try:
        # Setup test environment
        setup_test_users()

        # Run tests
        test_permission_checks()
        test_agent_creation()
        test_tool_creation()
        test_admin_creation()
        test_listing_entities()
        test_user_management()

        logger.info("=" * 50)
        logger.info("All tests completed successfully!")

        # Show final system state
        logger.info("Final system state:")
        creator = EntityCreator()
        entities = creator.list_entities("all")
        if entities["status"] == "success":
            agents = entities["result"].get("agents", [])
            tools = entities["result"].get("tools", [])
            logger.info(f"Total agents: {len(agents)}")
            logger.info(f"Total tools: {len(tools)}")

        users = role_manager.list_users()
        logger.info(f"Total users: {len(users)}")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
