#!/usr/bin/env python3
"""
Phase 1 Validation Script
Validates all Phase 1 components of the permissioned creation system.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any


class Phase1Validator:
    """Validates Phase 1 implementation components."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.successes = []

    def log_success(self, message: str):
        """Log a success message."""
        print(f"✅ {message}")
        self.successes.append(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        print(f"⚠️  {message}")
        self.warnings.append(message)

    def log_error(self, message: str):
        """Log an error message."""
        print(f"❌ {message}")
        self.errors.append(message)

    def check_file_exists(self, filepath: str, description: str) -> bool:
        """Check if a file exists."""
        if os.path.exists(filepath):
            self.log_success(f"{description}: {filepath}")
            return True
        else:
            self.log_error(f"{description}: {filepath} (missing)")
            return False

    def check_directory_exists(self, dirpath: str, description: str) -> bool:
        """Check if a directory exists."""
        if os.path.isdir(dirpath):
            self.log_success(f"{description}: {dirpath}")
            return True
        else:
            self.log_error(f"{description}: {dirpath} (missing)")
            return False

    def check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            self.log_success(
                f"Python version: {version.major}.{version.minor}.{version.micro}"
            )
            return True
        else:
            self.log_error(
                f"Python version: {version.major}.{version.minor}.{version.micro} (requires 3.11+)"
            )
            return False

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = [
            "langchain_core",
            "langgraph",
            "pydantic",
            "pytest",
            "black",
            "isort",
            "flake8",
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
                self.log_success(f"Dependency: {package}")
            except ImportError:
                missing.append(package)
                self.log_error(f"Dependency: {package} (missing)")

        return len(missing) == 0

    def check_git_hook(self) -> bool:
        """Check if Git pre-commit hook is properly configured."""
        hook_path = ".git/hooks/pre-commit"

        if not os.path.exists(hook_path):
            self.log_error("Git pre-commit hook: missing")
            return False

        # Check if executable
        if not os.access(hook_path, os.X_OK):
            self.log_warning(
                "Git pre-commit hook: not executable (run: chmod +x .git/hooks/pre-commit)"
            )
            return False

        self.log_success("Git pre-commit hook: configured and executable")
        return True

    def check_github_actions(self) -> bool:
        """Check if GitHub Actions workflow is configured."""
        workflow_path = ".github/workflows/ci-cd.yml"
        return self.check_file_exists(workflow_path, "GitHub Actions workflow")

    def check_vscode_config(self) -> bool:
        """Check if VS Code configuration is present."""
        settings_path = ".vscode/settings.json"
        extensions_path = ".vscode/extensions.json"

        settings_ok = self.check_file_exists(settings_path, "VS Code settings")
        extensions_ok = self.check_file_exists(extensions_path, "VS Code extensions")

        return settings_ok and extensions_ok

    def check_documentation(self) -> bool:
        """Check if documentation files are present."""
        docs = [
            ("docs/ONBOARDING_GUIDE.md", "Onboarding guide"),
            ("SETUP_INSTRUCTIONS.md", "Setup instructions"),
            ("PHASE1_IMPLEMENTATION.md", "Phase 1 implementation guide"),
            ("PERMISSIONED_CREATION_SYSTEM.md", "System architecture"),
            ("IMPLEMENTATION_ROADMAP.md", "Implementation roadmap"),
        ]

        all_ok = True
        for filepath, description in docs:
            if not self.check_file_exists(filepath, description):
                all_ok = False

        return all_ok

    def check_project_structure(self) -> bool:
        """Check if project structure is correct."""
        required_dirs = [
            ("agents", "Agents directory"),
            ("tools", "Tools directory"),
            ("tests", "Tests directory"),
            ("tests/agents", "Agent tests directory"),
            ("tests/tools", "Tool tests directory"),
            ("core", "Core modules directory"),
            ("templates", "Templates directory"),
            ("docs", "Documentation directory"),
            ("docs/agents", "Agent documentation directory"),
            ("docs/tools", "Tool documentation directory"),
        ]

        all_ok = True
        for dirpath, description in required_dirs:
            if not self.check_directory_exists(dirpath, description):
                all_ok = False

        return all_ok

    def check_core_files(self) -> bool:
        """Check if core system files are present."""
        core_files = [
            ("create_entity.py", "Entity creator"),
            ("core/roles.py", "Role management"),
            ("templates/agent_template.py", "Agent template"),
            ("templates/tool_template.py", "Tool template"),
            ("test_permissioned_system.py", "System tests"),
            ("setup_permissioned_system.py", "Setup script"),
        ]

        all_ok = True
        for filepath, description in core_files:
            if not self.check_file_exists(filepath, description):
                all_ok = False

        return all_ok

    def check_requirements_txt(self) -> bool:
        """Check if requirements.txt is properly formatted."""
        if not self.check_file_exists("requirements.txt", "Requirements file"):
            return False

        try:
            with open("requirements.txt", "r") as f:
                content = f.read()

            # Check for basic structure
            if "langchain-core" in content and "pytest" in content:
                self.log_success("Requirements.txt: properly formatted")
                return True
            else:
                self.log_error("Requirements.txt: missing core dependencies")
                return False

        except Exception as e:
            self.log_error(f"Requirements.txt: error reading file - {e}")
            return False

    def test_permissioned_system(self) -> bool:
        """Test the permissioned system functionality."""
        try:
            result = subprocess.run(
                [sys.executable, "test_permissioned_system.py"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.log_success("Permissioned system: tests passed")
                return True
            else:
                self.log_error(f"Permissioned system: tests failed - {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.log_error("Permissioned system: tests timed out")
            return False
        except Exception as e:
            self.log_error(f"Permissioned system: error running tests - {e}")
            return False

    def test_git_hook(self) -> bool:
        """Test the Git pre-commit hook."""
        try:
            # Create a test file
            test_file = "test_validation.py"
            with open(test_file, "w") as f:
                f.write("# Test file for validation\n")

            # Add to git
            subprocess.run(["git", "add", test_file], check=True)

            # Try to commit (should trigger hook)
            result = subprocess.run(
                ["git", "commit", "-m", "Test commit for validation"],
                capture_output=True,
                text=True,
            )

            # Clean up - handle case where there's no commit history
            try:
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
            except subprocess.CalledProcessError:
                # No commit history, just remove the file
                subprocess.run(["git", "rm", "--cached", test_file], check=True)

            if os.path.exists(test_file):
                os.remove(test_file)

            if result.returncode == 0:
                self.log_success("Git hook: validation working")
                return True
            else:
                self.log_warning(f"Git hook: validation failed - {result.stderr}")
                return False

        except Exception as e:
            self.log_error(f"Git hook: error testing - {e}")
            return False

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("🔍 Starting Phase 1 validation...\n")

        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Project Structure", self.check_project_structure),
            ("Core Files", self.check_core_files),
            ("Requirements", self.check_requirements_txt),
            ("Git Hook", self.check_git_hook),
            ("GitHub Actions", self.check_github_actions),
            ("VS Code Config", self.check_vscode_config),
            ("Documentation", self.check_documentation),
            ("Permissioned System", self.test_permissioned_system),
            ("Git Hook Test", self.test_git_hook),
        ]

        results = {}
        for name, check_func in checks:
            print(f"\n📋 {name}:")
            try:
                results[name] = check_func()
            except Exception as e:
                self.log_error(f"{name}: unexpected error - {e}")
                results[name] = False

        return results

    def generate_report(self, results: Dict[str, bool]) -> bool:
        """Generate a validation report."""
        print("\n" + "=" * 60)
        print("📊 PHASE 1 VALIDATION REPORT")
        print("=" * 60)

        total_checks = len(results)
        passed_checks = sum(results.values())
        failed_checks = total_checks - passed_checks

        print(f"\n📈 Summary:")
        print(f"   Total checks: {total_checks}")
        print(f"   ✅ Passed: {passed_checks}")
        print(f"   ❌ Failed: {failed_checks}")
        print(f"   ⚠️  Warnings: {len(self.warnings)}")

        if failed_checks == 0:
            print(f"\n🎉 Phase 1 validation PASSED!")
            print("   All components are properly configured.")
        else:
            print(f"\n⚠️  Phase 1 validation has issues:")
            print("   Please fix the errors above before proceeding.")

        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")

        if self.errors:
            print(f"\n❌ Errors:")
            for error in self.errors:
                print(f"   - {error}")

        print(f"\n📋 Detailed Results:")
        for name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {name}: {status}")

        print("\n" + "=" * 60)

        # Return appropriate exit code
        return failed_checks == 0


def main():
    """Main validation function."""
    validator = Phase1Validator()

    try:
        results = validator.run_all_checks()
        success = validator.generate_report(results)

        if success:
            print("\n🚀 Phase 1 is ready for team onboarding!")
            print("   Next steps:")
            print("   1. Share setup instructions with team")
            print("   2. Create user accounts")
            print("   3. Begin training sessions")
            sys.exit(0)
        else:
            print("\n🔧 Please fix the issues above before proceeding.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
