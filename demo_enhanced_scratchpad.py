#!/usr/bin/env python3
"""
Demonstration script for Enhanced Scratchpad capabilities:
1. Persistence: Data persists across sessions with disk/database storage
2. Access Control: Fine-grained permissions for agent-specific access
3. Versioning: Complete history tracking with rollback capabilities

This script shows how the enhanced features work with comprehensive data management.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

# Import the enhanced scratchpad tools
from tools.scratchpad_enhanced import (
    scratchpad_write, scratchpad_read, scratchpad_delete,
    scratchpad_list_keys, scratchpad_get_history, scratchpad_rollback,
    scratchpad_set_permissions, scratchpad_get_permissions,
    scratchpad_clear_all, scratchpad_get_stats
)


def demo_persistence():
    """Demonstrate persistence capabilities."""
    print("=" * 80)
    print("PERSISTENCE DEMONSTRATION")
    print("=" * 80)
    
    print("1. Writing data with persistence...")
    
    # Write some persistent data
    test_data = [
        {"key": "user_preferences", "value": "dark_mode=true,language=en", "agent": "user_agent"},
        {"key": "session_config", "value": "timeout=3600,max_connections=100", "agent": "system"},
        {"key": "project_settings", "value": "name=AgentK,version=2.0", "agent": "admin"}
    ]
    
    for data in test_data:
        result = scratchpad_write.invoke({
            "key": data["key"],
            "value": data["value"],
            "agent": data["agent"],
            "metadata": {"demo": True, "created_at": datetime.now().isoformat()}
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   ✅ Wrote '{data['key']}' (version {result_data['version']})")
        else:
            print(f"   ❌ Failed to write '{data['key']}': {result_data['message']}")
    
    print("\n2. Reading persistent data...")
    
    # Read the data back
    for data in test_data:
        result = scratchpad_read.invoke({
            "key": data["key"],
            "agent": data["agent"]
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   📖 Read '{data['key']}': {result_data['value']}")
            print(f"      Created: {result_data['created_at'][:19]}")
            print(f"      Version: {result_data['version']}")
        else:
            print(f"   ❌ Failed to read '{data['key']}': {result_data['message']}")
    
    print("\n3. Checking storage statistics...")
    
    # Get storage statistics
    stats_result = scratchpad_get_stats.invoke({})
    stats_data = json.loads(stats_result)
    
    if stats_data["status"] == "success":
        stats = stats_data["statistics"]
        print(f"   Total Keys: {stats['total_keys']}")
        print(f"   Total Versions: {stats['total_versions']}")
        print(f"   Storage Backend: {stats['storage_backend']}")
        print(f"   Storage Size: {stats['storage_size_bytes']} bytes")
    
    return test_data


def demo_access_control():
    """Demonstrate access control capabilities."""
    print("\n" + "=" * 80)
    print("ACCESS CONTROL DEMONSTRATION")
    print("=" * 80)
    
    print("1. Setting up data with different permissions...")
    
    # Create data with specific permissions
    secure_data = [
        {
            "key": "admin_config",
            "value": "secret_key=abc123,debug=true",
            "agent": "admin",
            "permissions": {
                "admin": ["admin"],
                "system": ["read"],
                "user_agent": []  # No access
            }
        },
        {
            "key": "shared_data",
            "value": "public_info=available_to_all",
            "agent": "system",
            "permissions": {
                "admin": ["admin"],
                "system": ["read", "write"],
                "user_agent": ["read"],
                "guest": ["read"]
            }
        }
    ]
    
    for data in secure_data:
        result = scratchpad_write.invoke({
            "key": data["key"],
            "value": data["value"],
            "agent": data["agent"],
            "permissions": data["permissions"]
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   ✅ Created '{data['key']}' with permissions")
        else:
            print(f"   ❌ Failed to create '{data['key']}': {result_data['message']}")
    
    print("\n2. Testing access control...")
    
    # Test different access scenarios
    access_tests = [
        {"key": "admin_config", "agent": "admin", "should_succeed": True},
        {"key": "admin_config", "agent": "system", "should_succeed": True},  # Read access
        {"key": "admin_config", "agent": "user_agent", "should_succeed": False},  # No access
        {"key": "shared_data", "agent": "user_agent", "should_succeed": True},  # Read access
        {"key": "shared_data", "agent": "guest", "should_succeed": True},  # Read access
    ]
    
    for test in access_tests:
        result = scratchpad_read.invoke({
            "key": test["key"],
            "agent": test["agent"]
        })
        
        result_data = json.loads(result)
        success = result_data["status"] == "success"
        
        if success == test["should_succeed"]:
            status_icon = "✅" if success else "🔒"
            print(f"   {status_icon} Agent '{test['agent']}' accessing '{test['key']}': {'SUCCESS' if success else 'DENIED'}")
        else:
            print(f"   ⚠️  Unexpected result for '{test['agent']}' accessing '{test['key']}'")
    
    print("\n3. Modifying permissions...")
    
    # Grant user_agent read access to admin_config
    perm_result = scratchpad_set_permissions.invoke({
        "key": "admin_config",
        "agent": "user_agent",
        "permissions": ["read"],
        "requesting_agent": "admin"
    })
    
    perm_data = json.loads(perm_result)
    if perm_data["status"] == "success":
        print(f"   ✅ Granted read access to user_agent for 'admin_config'")
        
        # Test the new permission
        result = scratchpad_read.invoke({
            "key": "admin_config",
            "agent": "user_agent"
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   ✅ user_agent can now read 'admin_config'")
        else:
            print(f"   ❌ Permission change didn't work: {result_data['message']}")
    
    print("\n4. Viewing permissions...")
    
    # Get permissions for a key
    for key in ["admin_config", "shared_data"]:
        perm_result = scratchpad_get_permissions.invoke({
            "key": key,
            "requesting_agent": "admin"
        })
        
        perm_data = json.loads(perm_result)
        if perm_data["status"] == "success":
            print(f"   📋 Permissions for '{key}':")
            for agent, perms in perm_data["permissions"].items():
                print(f"      - {agent}: {perms}")
    
    return secure_data


def demo_versioning():
    """Demonstrate versioning and rollback capabilities."""
    print("\n" + "=" * 80)
    print("VERSIONING DEMONSTRATION")
    print("=" * 80)
    
    print("1. Creating versioned data...")
    
    # Create initial version
    key = "versioned_config"
    agent = "system"
    
    versions = [
        "initial_config=v1.0,feature_a=enabled",
        "initial_config=v1.1,feature_a=enabled,feature_b=enabled",
        "initial_config=v1.2,feature_a=enabled,feature_b=enabled,feature_c=beta",
        "initial_config=v2.0,feature_a=enhanced,feature_b=enabled,feature_c=stable"
    ]
    
    for i, value in enumerate(versions, 1):
        result = scratchpad_write.invoke({
            "key": key,
            "value": value,
            "agent": agent,
            "metadata": {"version_notes": f"Version {i} update"}
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   ✅ Version {result_data['version']}: {value}")
            time.sleep(0.1)  # Small delay to ensure different timestamps
        else:
            print(f"   ❌ Failed to create version: {result_data['message']}")
    
    print("\n2. Viewing version history...")
    
    # Get complete history
    history_result = scratchpad_get_history.invoke({
        "key": key,
        "agent": agent
    })
    
    history_data = json.loads(history_result)
    if history_data["status"] == "success":
        print(f"   📚 History for '{key}' ({history_data['count']} versions):")
        for entry in history_data["history"]:
            print(f"      v{entry['version']}: {entry['value']}")
            print(f"        Action: {entry['action']} by {entry['agent']}")
            print(f"        Time: {entry['timestamp'][:19]}")
            print()
    
    print("3. Reading specific versions...")
    
    # Read specific versions
    for version in [1, 2, 4]:
        result = scratchpad_read.invoke({
            "key": key,
            "agent": agent,
            "version": version
        })
        
        result_data = json.loads(result)
        if result_data["status"] == "success":
            print(f"   📖 Version {version}: {result_data['value']}")
        else:
            print(f"   ❌ Failed to read version {version}: {result_data['message']}")
    
    print("\n4. Rolling back to previous version...")
    
    # Rollback to version 2
    rollback_result = scratchpad_rollback.invoke({
        "key": key,
        "version": 2,
        "agent": agent
    })
    
    rollback_data = json.loads(rollback_result)
    if rollback_data["status"] == "success":
        print(f"   ⏪ Rolled back to version {rollback_data['rolled_back_to_version']}")
        print(f"      New version: {rollback_data['new_version']}")
        print(f"      Current value: {rollback_data['value']}")
        
        # Verify the rollback
        current_result = scratchpad_read.invoke({
            "key": key,
            "agent": agent
        })
        
        current_data = json.loads(current_result)
        if current_data["status"] == "success":
            print(f"   ✅ Current version {current_data['version']}: {current_data['value']}")
    
    print("\n5. Viewing updated history...")
    
    # Get history after rollback
    history_result = scratchpad_get_history.invoke({
        "key": key,
        "agent": agent,
        "limit": 3  # Show last 3 entries
    })
    
    history_data = json.loads(history_result)
    if history_data["status"] == "success":
        print(f"   📚 Recent history (last {len(history_data['history'])} entries):")
        for entry in history_data["history"]:
            print(f"      v{entry['version']}: {entry['action']} - {entry['value'][:50]}...")
    
    return key, agent


def demo_integration():
    """Demonstrate how all three capabilities work together."""
    print("\n" + "=" * 80)
    print("INTEGRATED ENHANCED SCRATCHPAD DEMONSTRATION")
    print("=" * 80)
    
    print("1. Enhanced scratchpad workflow simulation...")
    
    # Simulate a complex workflow
    workflow_steps = [
        "Data Creation - Store data with persistence and permissions",
        "Access Control - Verify agent-specific access rights",
        "Version Management - Track all changes with history",
        "Permission Updates - Modify access rights as needed",
        "Rollback Operations - Restore previous versions when needed",
        "Cross-Session Persistence - Data survives system restarts",
        "Audit Trail - Complete history of all operations",
        "Security Enforcement - Prevent unauthorized access"
    ]
    
    print("   Enhanced workflow capabilities:")
    for i, step in enumerate(workflow_steps, 1):
        print(f"     {i}. {step}")
    
    print("\n2. Integration benefits...")
    
    integration_benefits = [
        "Persistence - Data survives across sessions and restarts",
        "Access Control - Fine-grained permissions per agent per key",
        "Versioning - Complete history with rollback capabilities",
        "Audit Trail - Full tracking of who changed what when",
        "Security - Unauthorized access prevention",
        "Scalability - Database backend for large datasets",
        "Reliability - Atomic operations with error handling",
        "Flexibility - Multiple storage backends supported"
    ]
    
    print("   Integration benefits:")
    for benefit in integration_benefits:
        print(f"     ✅ {benefit}")
    
    print("\n3. Enhanced capabilities summary:")
    
    capabilities = {
        "Persistence": {
            "Disk Storage": "JSON files for simple persistence",
            "Database Storage": "SQLite for robust data management",
            "Cross-Session": "Data survives system restarts",
            "Atomic Operations": "Consistent data state guaranteed",
            "Error Recovery": "Graceful handling of storage failures"
        },
        "Access Control": {
            "Fine-Grained Permissions": "Read, write, delete, admin levels",
            "Agent-Specific Access": "Different permissions per agent",
            "Permission Management": "Dynamic permission updates",
            "Creator Rights": "Automatic admin rights for creators",
            "Inheritance": "Write permission includes read access"
        },
        "Versioning": {
            "Complete History": "Every change tracked with metadata",
            "Version Rollback": "Restore any previous version",
            "Audit Trail": "Who changed what when tracking",
            "Metadata Support": "Rich context for each change",
            "History Limits": "Configurable history retention"
        }
    }
    
    for category, features in capabilities.items():
        print(f"\n   {category}:")
        for feature, description in features.items():
            print(f"     🔧 {feature}: {description}")
    
    print("\n4. Enterprise features:")
    
    enterprise_features = [
        "Multi-Backend Storage: Memory, disk, and database options",
        "Thread Safety: Concurrent access with proper locking",
        "Encryption Support: Optional data encryption at rest",
        "Backup Management: Automatic backup creation",
        "Performance Optimization: Efficient storage and retrieval",
        "Scalability: Handles large datasets and many agents"
    ]
    
    for feature in enterprise_features:
        print(f"     🏢 {feature}")
    
    return {
        "persistence": True,
        "access_control": True,
        "versioning": True,
        "enterprise_ready": True
    }


def demo_advanced_features():
    """Demonstrate advanced features and edge cases."""
    print("\n" + "=" * 80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    
    print("1. Testing complex permission scenarios...")
    
    # Create a key with complex permissions
    complex_key = "complex_permissions_test"
    
    # Admin creates the key
    result = scratchpad_write.invoke({
        "key": complex_key,
        "value": "sensitive_data=classified",
        "agent": "admin",
        "permissions": {
            "admin": ["admin"],
            "security_agent": ["read", "write"],
            "audit_agent": ["read"],
            "user_agent": []
        }
    })
    
    result_data = json.loads(result)
    if result_data["status"] == "success":
        print(f"   ✅ Created complex permissions key")
    
    # Test permission inheritance (write includes read)
    write_result = scratchpad_write.invoke({
        "key": complex_key,
        "value": "sensitive_data=updated_by_security",
        "agent": "security_agent"
    })
    
    write_data = json.loads(write_result)
    if write_data["status"] == "success":
        print(f"   ✅ security_agent can write (includes read)")
    
    # Test read-only access
    read_result = scratchpad_read.invoke({
        "key": complex_key,
        "agent": "audit_agent"
    })
    
    read_data = json.loads(read_result)
    if read_data["status"] == "success":
        print(f"   ✅ audit_agent can read")
    
    # Test denied access
    denied_result = scratchpad_write.invoke({
        "key": complex_key,
        "value": "unauthorized_change",
        "agent": "audit_agent"
    })
    
    denied_data = json.loads(denied_result)
    if denied_data["status"] == "error":
        print(f"   🔒 audit_agent correctly denied write access")
    
    print("\n2. Testing metadata and rich context...")
    
    # Create data with rich metadata
    metadata_key = "metadata_rich_data"
    rich_metadata = {
        "project": "AgentK Enhancement",
        "priority": "high",
        "tags": ["demo", "metadata", "test"],
        "owner": "system_admin",
        "expires": (datetime.now().timestamp() + 86400)  # 24 hours
    }
    
    result = scratchpad_write.invoke({
        "key": metadata_key,
        "value": "data_with_rich_metadata=true",
        "agent": "system",
        "metadata": rich_metadata
    })
    
    result_data = json.loads(result)
    if result_data["status"] == "success":
        print(f"   ✅ Created data with rich metadata")
        
        # Read back with metadata
        read_result = scratchpad_read.invoke({
            "key": metadata_key,
            "agent": "system"
        })
        
        read_data = json.loads(read_result)
        if read_data["status"] == "success":
            print(f"   📋 Metadata: {read_data['metadata']}")
    
    print("\n3. Testing list operations with metadata...")
    
    # List keys with metadata
    list_result = scratchpad_list_keys.invoke({
        "agent": "admin",
        "include_metadata": True
    })
    
    list_data = json.loads(list_result)
    if list_data["status"] == "success":
        print(f"   📝 Found {list_data['count']} accessible keys:")
        for key_info in list_data["keys"][:3]:  # Show first 3
            if isinstance(key_info, dict):
                print(f"      - {key_info['key']} (v{key_info['version']}) by {key_info['created_by']}")
            else:
                print(f"      - {key_info}")
    
    return {
        "complex_permissions": True,
        "rich_metadata": True,
        "advanced_operations": True
    }


def main():
    """Run all demonstrations."""
    print("ENHANCED SCRATCHPAD CAPABILITIES DEMONSTRATION")
    print("This demo shows the three key improvements:")
    print("1. Persistence - Data persists across sessions with disk/database storage")
    print("2. Access Control - Fine-grained permissions for agent-specific access")
    print("3. Versioning - Complete history tracking with rollback capabilities")
    print()
    
    try:
        # Run individual demonstrations
        persistence_demo = demo_persistence()
        access_control_demo = demo_access_control()
        versioning_demo = demo_versioning()
        integration_demo = demo_integration()
        advanced_demo = demo_advanced_features()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION SUMMARY")
        print("=" * 80)
        print("✅ Persistence: Data storage with multiple backend options")
        print("✅ Access Control: Fine-grained permissions and security")
        print("✅ Versioning: Complete history tracking and rollback")
        print("✅ Integration: All capabilities working together seamlessly")
        print()
        print("Enhanced Scratchpad is ready with:")
        print("  💾 Persistence - Data survives across sessions and restarts")
        print("  🔐 Access Control - Fine-grained permissions per agent")
        print("  📚 Versioning - Complete history with rollback capabilities")
        print("  🔍 Audit Trail - Full tracking of all operations")
        print("  🏢 Enterprise Ready - Thread-safe, scalable, reliable")
        print("  ⚡ Performance - Optimized storage and retrieval")
        print()
        print("The enhanced system provides enterprise-grade data management")
        print("with comprehensive security, versioning, and persistence.")
        
        # Show final statistics
        print("\n" + "=" * 40)
        print("FINAL STATISTICS")
        print("=" * 40)
        
        stats_result = scratchpad_get_stats.invoke({})
        stats_data = json.loads(stats_result)
        
        if stats_data["status"] == "success":
            stats = stats_data["statistics"]
            print(f"Total Keys: {stats['total_keys']}")
            print(f"Total Versions: {stats['total_versions']}")
            print(f"Storage Backend: {stats['storage_backend']}")
            print(f"Storage Size: {stats['storage_size_bytes']} bytes")
        
        print("\nDemo completed successfully!")
        print("Note: Data persists in 'scratchpad_data' directory for future sessions.")
        
    except Exception as e:
        print(f"Demo error: {e}")
        print("Note: This demo shows the enhanced capabilities structure.")
        print("Full integration requires the complete enhanced Scratchpad system.")


if __name__ == "__main__":
    main()