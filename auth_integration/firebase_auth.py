"""
Firebase Authentication Integration

Provides Firebase authentication for the permissioned creation system,
allowing users to authenticate with Google accounts and other Firebase providers.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import firebase_admin
    from firebase_admin import auth, credentials, firestore
    from firebase_admin.exceptions import FirebaseError

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning(
        "Firebase Admin SDK not installed. Install with: pip install firebase-admin"
    )

from core.roles import role_manager, UserRole, log_audit_event

logger = logging.getLogger(__name__)


class FirebaseAuthIntegration:
    """Firebase authentication integration for permissioned creation system"""

    def __init__(self, config_file: str = "auth_integration/firebase_config.json"):
        self.config_file = config_file
        self.app = None
        self.db = None
        self.users_collection = "permissioned_system_users"

        if FIREBASE_AVAILABLE:
            self._initialize_firebase()
        else:
            logger.error("Firebase Admin SDK not available")

    def _initialize_firebase(self):
        """Initialize Firebase app and database"""
        try:
            if os.path.exists(self.config_file):
                # Initialize with service account
                cred = credentials.Certificate(self.config_file)
                self.app = firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                logger.info("Firebase initialized successfully")
            else:
                logger.warning(f"Firebase config file not found: {self.config_file}")
                logger.info(
                    "Using default Firebase app (GOOGLE_APPLICATION_CREDENTIALS)"
                )
                self.app = firebase_admin.initialize_app()
                self.db = firestore.client()

        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")

    def verify_id_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        """Verify Firebase ID token and return user info"""
        if not FIREBASE_AVAILABLE or not self.app:
            return None

        try:
            decoded_token = auth.verify_id_token(id_token)
            return {
                "uid": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "name": decoded_token.get("name"),
                "picture": decoded_token.get("picture"),
                "email_verified": decoded_token.get("email_verified", False),
            }
        except FirebaseError as e:
            logger.error(f"Firebase token verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def get_or_create_user(self, firebase_user: Dict[str, Any]) -> Dict[str, Any]:
        """Get existing user or create new user in permissioned system"""
        uid = firebase_user["uid"]
        email = firebase_user.get("email", "")

        # Check if user exists in permissioned system
        existing_role = role_manager.get_user_role(uid)

        if existing_role:
            # User exists, update last login
            self._update_user_login(uid)
            return {
                "user_id": uid,
                "email": email,
                "role": existing_role.value,
                "exists": True,
            }
        else:
            # Create new user with default role
            default_role = UserRole.VIEWER  # Most restrictive default
            role_manager.set_user_role(uid, default_role, created_by="firebase_auth")

            # Store additional user info in Firestore
            self._store_user_info(uid, firebase_user)

            log_audit_event(
                uid,
                "user_created",
                "firebase_auth",
                {"email": email, "role": default_role.value, "provider": "firebase"},
            )

            return {
                "user_id": uid,
                "email": email,
                "role": default_role.value,
                "exists": False,
            }

    def _store_user_info(self, uid: str, firebase_user: Dict[str, Any]):
        """Store user information in Firestore"""
        if not self.db:
            return

        try:
            user_doc = {
                "uid": uid,
                "email": firebase_user.get("email"),
                "name": firebase_user.get("name"),
                "picture": firebase_user.get("picture"),
                "email_verified": firebase_user.get("email_verified", False),
                "created_at": datetime.now().isoformat(),
                "last_login": datetime.now().isoformat(),
                "provider": "firebase",
            }

            self.db.collection(self.users_collection).document(uid).set(user_doc)

        except Exception as e:
            logger.error(f"Failed to store user info in Firestore: {e}")

    def _update_user_login(self, uid: str):
        """Update user's last login time"""
        if not self.db:
            return

        try:
            self.db.collection(self.users_collection).document(uid).update(
                {"last_login": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Failed to update user login time: {e}")

    def get_user_info(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get user information from Firestore"""
        if not self.db:
            return None

        try:
            doc = self.db.collection(self.users_collection).document(uid).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")

        return None

    def update_user_role(self, uid: str, new_role: UserRole, updated_by: str):
        """Update user role in permissioned system"""
        try:
            role_manager.set_user_role(uid, new_role, created_by=updated_by)

            # Update in Firestore
            if self.db:
                self.db.collection(self.users_collection).document(uid).update(
                    {
                        "role": new_role.value,
                        "role_updated_at": datetime.now().isoformat(),
                        "role_updated_by": updated_by,
                    }
                )

            log_audit_event(
                updated_by,
                "role_updated",
                uid,
                {"new_role": new_role.value, "target_user": uid},
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update user role: {e}")
            return False

    def list_users(self) -> Dict[str, Any]:
        """List all users from Firestore"""
        if not self.db:
            return {"users": []}

        try:
            users = []
            docs = self.db.collection(self.users_collection).stream()

            for doc in docs:
                user_data = doc.to_dict()
                user_data["uid"] = doc.id

                # Get role from permissioned system
                role = role_manager.get_user_role(doc.id)
                user_data["role"] = role.value if role else "unknown"

                users.append(user_data)

            return {"users": users}

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return {"users": [], "error": str(e)}

    def delete_user(self, uid: str, deleted_by: str) -> bool:
        """Delete user from both systems"""
        try:
            # Delete from permissioned system
            role_manager.delete_user(uid)

            # Delete from Firestore
            if self.db:
                self.db.collection(self.users_collection).document(uid).delete()

            log_audit_event(deleted_by, "user_deleted", uid, {"deleted_user": uid})

            return True

        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            return False

    def get_user_activity(self, uid: str, days: int = 7) -> Dict[str, Any]:
        """Get user activity from audit logs"""
        try:
            # Read audit log file
            audit_file = "audit.log"
            user_logs = []

            if os.path.exists(audit_file):
                cutoff_date = datetime.now() - timedelta(days=days)

                with open(audit_file, "r") as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            if log_entry.get("user_id") == uid:
                                log_date = datetime.fromisoformat(
                                    log_entry["timestamp"]
                                )
                                if log_date >= cutoff_date:
                                    user_logs.append(log_entry)
                        except json.JSONDecodeError:
                            continue

            return {
                "user_id": uid,
                "activity_logs": user_logs,
                "total_actions": len(user_logs),
            }

        except Exception as e:
            logger.error(f"Failed to get user activity: {e}")
            return {"user_id": uid, "activity_logs": [], "total_actions": 0}


class FirebaseAuthMiddleware:
    """Middleware for Firebase authentication"""

    def __init__(self, firebase_auth: FirebaseAuthIntegration):
        self.firebase_auth = firebase_auth

    def authenticate_request(
        self, request_headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Authenticate request using Firebase token"""
        auth_header = request_headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return None

        id_token = auth_header.split("Bearer ")[1]
        firebase_user = self.firebase_auth.verify_id_token(id_token)

        if not firebase_user:
            return None

        # Get or create user in permissioned system
        user_info = self.firebase_auth.get_or_create_user(firebase_user)
        return user_info


# Example configuration file structure
FIREBASE_CONFIG_EXAMPLE = {
    "type": "service_account",
    "project_id": "your-project-id",
    "private_key_id": "your-private-key-id",
    "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com",
    "client_id": "your-client-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-xxxxx%40your-project-id.iam.gserviceaccount.com",
}


def create_firebase_config_template():
    """Create a template Firebase configuration file"""
    config_dir = os.path.dirname("auth_integration/firebase_config.json")
    os.makedirs(config_dir, exist_ok=True)

    template_path = "auth_integration/firebase_config_template.json"
    with open(template_path, "w") as f:
        json.dump(FIREBASE_CONFIG_EXAMPLE, f, indent=2)

    logger.info(f"Firebase config template created: {template_path}")
    logger.info("Please update with your actual Firebase service account credentials")


# Integration with Flask app
def create_firebase_auth_blueprint(firebase_auth: FirebaseAuthIntegration):
    """Create Flask blueprint for Firebase authentication"""
    from flask import Blueprint, request, jsonify, session

    auth_bp = Blueprint("firebase_auth", __name__)

    @auth_bp.route("/auth/firebase/login", methods=["POST"])
    def firebase_login():
        """Firebase login endpoint"""
        try:
            data = request.get_json()
            id_token = data.get("idToken")

            if not id_token:
                return jsonify({"error": "ID token required"}), 400

            # Verify token and get user
            firebase_user = firebase_auth.verify_id_token(id_token)
            if not firebase_user:
                return jsonify({"error": "Invalid token"}), 401

            # Get or create user in permissioned system
            user_info = firebase_auth.get_or_create_user(firebase_user)

            # Set session
            session["user_id"] = user_info["user_id"]
            session["firebase_user"] = firebase_user

            log_audit_event(
                user_info["user_id"],
                "login",
                "firebase_auth",
                {"email": firebase_user.get("email"), "provider": "firebase"},
            )

            return jsonify(
                {"success": True, "user": user_info, "firebase_user": firebase_user}
            )

        except Exception as e:
            logger.error(f"Firebase login error: {e}")
            return jsonify({"error": "Authentication failed"}), 500

    @auth_bp.route("/auth/firebase/logout", methods=["POST"])
    def firebase_logout():
        """Firebase logout endpoint"""
        if "user_id" in session:
            log_audit_event(
                session["user_id"], "logout", "firebase_auth", {"provider": "firebase"}
            )
            session.clear()

        return jsonify({"success": True})

    @auth_bp.route("/auth/firebase/user", methods=["GET"])
    def get_firebase_user():
        """Get current Firebase user info"""
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401

        user_id = session["user_id"]
        user_info = firebase_auth.get_user_info(user_id)

        if user_info:
            role = role_manager.get_user_role(user_id)
            user_info["role"] = role.value if role else "unknown"
            return jsonify(user_info)
        else:
            return jsonify({"error": "User not found"}), 404

    return auth_bp


if __name__ == "__main__":
    # Create config template if it doesn't exist
    create_firebase_config_template()

    # Test Firebase integration
    if FIREBASE_AVAILABLE:
        firebase_auth = FirebaseAuthIntegration()
        print("Firebase integration ready")
    else:
        print("Firebase Admin SDK not available")
        print("Install with: pip install firebase-admin")
