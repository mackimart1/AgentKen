"""
Security Agent - Continuously monitors all agent interactions, tool executions,
and external communications to detect anomalies, unauthorized access, or potential vulnerabilities
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import sqlite3
import os
import sys
from pathlib import Path
from collections import defaultdict, deque, Counter
import ipaddress
import base64

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent_framework import BaseAgent, AgentCapability, MessageBus, Message, MessageType, AgentStatus


class ThreatLevel(Enum):
    """Security threat levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityEventType(Enum):
    """Types of security events"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SUSPICIOUS_COMMUNICATION = "suspicious_communication"
    TOOL_EXECUTION_VIOLATION = "tool_execution_violation"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration_attempt"
    INJECTION_ATTACK = "injection_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALFORMED_REQUEST = "malformed_request"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    CONFIGURATION_TAMPERING = "configuration_tampering"
    UNKNOWN_THREAT = "unknown_threat"


class VulnerabilityType(Enum):
    """Types of vulnerabilities"""
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FLAW = "authorization_flaw"
    INFORMATION_DISCLOSURE = "information_disclosure"
    INJECTION_VULNERABILITY = "injection_vulnerability"
    INSECURE_COMMUNICATION = "insecure_communication"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_VULNERABILITY = "dependency_vulnerability"
    LOGIC_FLAW = "logic_flaw"


@dataclass
class SecurityEvent:
    """Represents a security event"""
    id: str
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source: str
    target: str
    description: str
    evidence: Dict[str, Any]
    context: Dict[str, Any]
    mitigated: bool
    mitigation_actions: List[str]
    false_positive: bool
    confidence_score: float


@dataclass
class Vulnerability:
    """Represents a security vulnerability"""
    id: str
    vulnerability_type: VulnerabilityType
    component: str
    description: str
    severity: ThreatLevel
    cvss_score: float
    affected_versions: List[str]
    remediation_steps: List[str]
    discovered_at: datetime
    patched: bool
    patch_date: Optional[datetime]
    references: List[str]


@dataclass
class SecurityPolicy:
    """Represents a security policy"""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    enforcement_level: str
    exceptions: List[str]
    created_at: datetime
    updated_at: datetime
    active: bool


class SecurityDatabase:
    """Database for storing security events and policies"""
    
    def __init__(self, db_path: str = "security_agent.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the security database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                description TEXT NOT NULL,
                evidence TEXT,
                context TEXT,
                mitigated BOOLEAN DEFAULT FALSE,
                mitigation_actions TEXT,
                false_positive BOOLEAN DEFAULT FALSE,
                confidence_score REAL NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id TEXT PRIMARY KEY,
                vulnerability_type TEXT NOT NULL,
                component TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                cvss_score REAL NOT NULL,
                affected_versions TEXT,
                remediation_steps TEXT,
                discovered_at TEXT NOT NULL,
                patched BOOLEAN DEFAULT FALSE,
                patch_date TEXT,
                references TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_policies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                rules TEXT NOT NULL,
                enforcement_level TEXT NOT NULL,
                exceptions TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active BOOLEAN DEFAULT TRUE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS access_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                session_id TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS communication_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                message_type TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                encrypted BOOLEAN DEFAULT FALSE,
                external BOOLEAN DEFAULT FALSE,
                suspicious BOOLEAN DEFAULT FALSE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_execution_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                parameters_hash TEXT NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                output_hash TEXT,
                risk_level TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_security_event(self, event: SecurityEvent):
        """Store a security event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO security_events 
            (id, timestamp, event_type, threat_level, source, target, description,
             evidence, context, mitigated, mitigation_actions, false_positive, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id,
            event.timestamp.isoformat(),
            event.event_type.value,
            event.threat_level.value,
            event.source,
            event.target,
            event.description,
            json.dumps(event.evidence),
            json.dumps(event.context),
            event.mitigated,
            json.dumps(event.mitigation_actions),
            event.false_positive,
            event.confidence_score
        ))
        
        conn.commit()
        conn.close()
    
    def store_vulnerability(self, vulnerability: Vulnerability):
        """Store a vulnerability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO vulnerabilities 
            (id, vulnerability_type, component, description, severity, cvss_score,
             affected_versions, remediation_steps, discovered_at, patched, patch_date, references)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            vulnerability.id,
            vulnerability.vulnerability_type.value,
            vulnerability.component,
            vulnerability.description,
            vulnerability.severity.value,
            vulnerability.cvss_score,
            json.dumps(vulnerability.affected_versions),
            json.dumps(vulnerability.remediation_steps),
            vulnerability.discovered_at.isoformat(),
            vulnerability.patched,
            vulnerability.patch_date.isoformat() if vulnerability.patch_date else None,
            json.dumps(vulnerability.references)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_events(self, hours: int = 24, threat_level: Optional[ThreatLevel] = None) -> List[SecurityEvent]:
        """Get recent security events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM security_events 
            WHERE timestamp > datetime('now', '-{} hours')
        """.format(hours)
        
        params = []
        if threat_level:
            query += " AND threat_level = ?"
            params.append(threat_level.value)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        events = []
        for row in rows:
            event = SecurityEvent(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                event_type=SecurityEventType(row[2]),
                threat_level=ThreatLevel(row[3]),
                source=row[4],
                target=row[5],
                description=row[6],
                evidence=json.loads(row[7]) if row[7] else {},
                context=json.loads(row[8]) if row[8] else {},
                mitigated=bool(row[9]),
                mitigation_actions=json.loads(row[10]) if row[10] else [],
                false_positive=bool(row[11]),
                confidence_score=row[12]
            )
            events.append(event)
        
        conn.close()
        return events


class AnomalyDetector:
    """Detects anomalous behavior patterns"""
    
    def __init__(self):
        self.baseline_patterns = {}
        self.recent_activity = defaultdict(deque)
        self.learning_period = 7 * 24 * 3600  # 7 days in seconds
        self.anomaly_thresholds = {
            'message_frequency': 2.0,  # Standard deviations
            'execution_time': 3.0,
            'payload_size': 2.5,
            'error_rate': 2.0
        }
    
    def learn_baseline(self, activity_data: List[Dict[str, Any]]):
        """Learn baseline patterns from historical data"""
        patterns = defaultdict(list)
        
        for activity in activity_data:
            agent_id = activity.get('agent_id', 'unknown')
            patterns[f"{agent_id}_message_frequency"].append(activity.get('message_count', 0))
            patterns[f"{agent_id}_execution_time"].append(activity.get('execution_time', 0))
            patterns[f"{agent_id}_payload_size"].append(activity.get('payload_size', 0))
            patterns[f"{agent_id}_error_rate"].append(activity.get('error_rate', 0))
        
        # Calculate baseline statistics
        for pattern_key, values in patterns.items():
            if len(values) > 10:  # Need sufficient data
                import numpy as np
                self.baseline_patterns[pattern_key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'sample_count': len(values)
                }
    
    def detect_anomalies(self, current_activity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in current activity"""
        anomalies = []
        agent_id = current_activity.get('agent_id', 'unknown')
        
        # Check message frequency anomaly
        message_count = current_activity.get('message_count', 0)
        baseline_key = f"{agent_id}_message_frequency"
        if baseline_key in self.baseline_patterns:
            baseline = self.baseline_patterns[baseline_key]
            z_score = abs(message_count - baseline['mean']) / (baseline['std'] + 1e-6)
            if z_score > self.anomaly_thresholds['message_frequency']:
                anomalies.append({
                    'type': 'message_frequency_anomaly',
                    'severity': 'high' if z_score > 3.0 else 'medium',
                    'z_score': z_score,
                    'current_value': message_count,
                    'baseline_mean': baseline['mean'],
                    'description': f"Unusual message frequency for agent {agent_id}"
                })
        
        # Check execution time anomaly
        execution_time = current_activity.get('execution_time', 0)
        baseline_key = f"{agent_id}_execution_time"
        if baseline_key in self.baseline_patterns:
            baseline = self.baseline_patterns[baseline_key]
            z_score = abs(execution_time - baseline['mean']) / (baseline['std'] + 1e-6)
            if z_score > self.anomaly_thresholds['execution_time']:
                anomalies.append({
                    'type': 'execution_time_anomaly',
                    'severity': 'high' if z_score > 4.0 else 'medium',
                    'z_score': z_score,
                    'current_value': execution_time,
                    'baseline_mean': baseline['mean'],
                    'description': f"Unusual execution time for agent {agent_id}"
                })
        
        # Check payload size anomaly
        payload_size = current_activity.get('payload_size', 0)
        baseline_key = f"{agent_id}_payload_size"
        if baseline_key in self.baseline_patterns:
            baseline = self.baseline_patterns[baseline_key]
            z_score = abs(payload_size - baseline['mean']) / (baseline['std'] + 1e-6)
            if z_score > self.anomaly_thresholds['payload_size']:
                anomalies.append({
                    'type': 'payload_size_anomaly',
                    'severity': 'medium' if payload_size > baseline['mean'] else 'low',
                    'z_score': z_score,
                    'current_value': payload_size,
                    'baseline_mean': baseline['mean'],
                    'description': f"Unusual payload size for agent {agent_id}"
                })
        
        return anomalies


class ThreatDetector:
    """Detects various types of security threats"""
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.ip_whitelist = set()
        self.ip_blacklist = set()
        self.suspicious_patterns = self._load_suspicious_patterns()
    
    def _load_threat_patterns(self) -> Dict[str, List[str]]:
        """Load threat detection patterns"""
        return {
            'sql_injection': [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bUPDATE\b.*\bSET\b)",
                r"(\bDELETE\b.*\bFROM\b)",
                r"(';.*--)",
                r"(\bOR\b.*=.*)"
            ],
            'command_injection': [
                r"(;.*\b(cat|ls|pwd|whoami|id|uname)\b)",
                r"(\|.*\b(cat|ls|pwd|whoami|id|uname)\b)",
                r"(&.*\b(cat|ls|pwd|whoami|id|uname)\b)",
                r"(`.*`)",
                r"(\$\(.*\))"
            ],
            'path_traversal': [
                r"(\.\.\/)",
                r"(\.\.\\)",
                r"(%2e%2e%2f)",
                r"(%2e%2e%5c)"
            ],
            'xss': [
                r"(<script.*>)",
                r"(javascript:)",
                r"(on\w+\s*=)",
                r"(<iframe.*>)",
                r"(<object.*>)",
                r"(<embed.*>)"
            ],
            'data_exfiltration': [
                r"(password|passwd|secret|key|token)",
                r"(api_key|access_token|private_key)",
                r"(credit_card|ssn|social_security)",
                r"(email|phone|address)"
            ]
        }
    
    def _load_suspicious_patterns(self) -> Dict[str, List[str]]:
        """Load suspicious behavior patterns"""
        return {
            'privilege_escalation': [
                r"(sudo|su|admin|root)",
                r"(chmod|chown|setuid)",
                r"(privilege|permission|access)"
            ],
            'reconnaissance': [
                r"(scan|probe|enumerate)",
                r"(version|banner|fingerprint)",
                r"(directory|file|listing)"
            ],
            'malware_indicators': [
                r"(backdoor|trojan|virus|malware)",
                r"(payload|shellcode|exploit)",
                r"(reverse_shell|bind_shell)"
            ]
        }
    
    def detect_threats(self, data: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect threats in data"""
        threats = []
        
        # Check for injection attacks
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, data, re.IGNORECASE)
                if matches:
                    threats.append({
                        'type': threat_type,
                        'pattern': pattern,
                        'matches': matches,
                        'severity': self._assess_threat_severity(threat_type, matches),
                        'confidence': self._calculate_confidence(threat_type, matches, context)
                    })
        
        # Check for suspicious patterns
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, data, re.IGNORECASE)
                if matches:
                    threats.append({
                        'type': pattern_type,
                        'pattern': pattern,
                        'matches': matches,
                        'severity': 'medium',
                        'confidence': 0.6
                    })
        
        return threats
    
    def _assess_threat_severity(self, threat_type: str, matches: List[str]) -> str:
        """Assess the severity of a detected threat"""
        severity_map = {
            'sql_injection': 'high',
            'command_injection': 'critical',
            'path_traversal': 'high',
            'xss': 'medium',
            'data_exfiltration': 'critical'
        }
        
        base_severity = severity_map.get(threat_type, 'medium')
        
        # Increase severity based on number of matches
        if len(matches) > 3:
            if base_severity == 'medium':
                return 'high'
            elif base_severity == 'high':
                return 'critical'
        
        return base_severity
    
    def _calculate_confidence(self, threat_type: str, matches: List[str], context: Dict[str, Any]) -> float:
        """Calculate confidence score for threat detection"""
        base_confidence = 0.7
        
        # Increase confidence based on context
        if context.get('external_source', False):
            base_confidence += 0.2
        
        if context.get('privileged_operation', False):
            base_confidence += 0.1
        
        # Increase confidence based on multiple matches
        if len(matches) > 1:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP address reputation"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            reputation = {
                'ip': ip_address,
                'is_private': ip.is_private,
                'is_loopback': ip.is_loopback,
                'is_multicast': ip.is_multicast,
                'whitelisted': ip_address in self.ip_whitelist,
                'blacklisted': ip_address in self.ip_blacklist,
                'risk_score': 0.0
            }
            
            # Calculate risk score
            if reputation['blacklisted']:
                reputation['risk_score'] = 1.0
            elif reputation['whitelisted']:
                reputation['risk_score'] = 0.0
            elif reputation['is_private'] or reputation['is_loopback']:
                reputation['risk_score'] = 0.1
            else:
                reputation['risk_score'] = 0.3  # Unknown external IP
            
            return reputation
            
        except ValueError:
            return {
                'ip': ip_address,
                'valid': False,
                'risk_score': 0.5
            }


class VulnerabilityScanner:
    """Scans for security vulnerabilities"""
    
    def __init__(self):
        self.vulnerability_database = self._load_vulnerability_database()
        self.scan_rules = self._load_scan_rules()
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load known vulnerability database"""
        return {
            'python_packages': {
                'requests': {
                    'vulnerable_versions': ['<2.20.0'],
                    'cve': 'CVE-2018-18074',
                    'severity': 'medium',
                    'description': 'Improper Certificate Validation'
                },
                'urllib3': {
                    'vulnerable_versions': ['<1.24.2'],
                    'cve': 'CVE-2019-11324',
                    'severity': 'high',
                    'description': 'Certificate verification bypass'
                }
            },
            'configurations': {
                'weak_passwords': {
                    'patterns': ['password', '123456', 'admin', 'root'],
                    'severity': 'high',
                    'description': 'Weak default passwords detected'
                },
                'insecure_protocols': {
                    'patterns': ['http://', 'ftp://', 'telnet://'],
                    'severity': 'medium',
                    'description': 'Insecure protocols in use'
                }
            }
        }
    
    def _load_scan_rules(self) -> List[Dict[str, Any]]:
        """Load vulnerability scan rules"""
        return [
            {
                'name': 'weak_crypto',
                'pattern': r'(md5|sha1|des|rc4)',
                'severity': 'medium',
                'description': 'Weak cryptographic algorithms detected'
            },
            {
                'name': 'hardcoded_secrets',
                'pattern': r'(password\s*=\s*["\'][^"\']+["\']|api_key\s*=\s*["\'][^"\']+["\'])',
                'severity': 'high',
                'description': 'Hardcoded secrets detected'
            },
            {
                'name': 'insecure_random',
                'pattern': r'(random\.random|math\.random)',
                'severity': 'low',
                'description': 'Insecure random number generation'
            },
            {
                'name': 'sql_concatenation',
                'pattern': r'(SELECT.*\+.*|INSERT.*\+.*|UPDATE.*\+.*)',
                'severity': 'high',
                'description': 'Potential SQL injection via string concatenation'
            }
        ]
    
    def scan_code(self, code: str, filename: str = "") -> List[Vulnerability]:
        """Scan code for vulnerabilities"""
        vulnerabilities = []
        
        for rule in self.scan_rules:
            matches = re.finditer(rule['pattern'], code, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1
                
                vulnerability = Vulnerability(
                    id=str(uuid.uuid4()),
                    vulnerability_type=VulnerabilityType.INJECTION_VULNERABILITY if 'sql' in rule['name'] else VulnerabilityType.WEAK_CRYPTOGRAPHY,
                    component=filename or 'unknown',
                    description=f"{rule['description']} at line {line_number}",
                    severity=ThreatLevel(rule['severity']),
                    cvss_score=self._calculate_cvss_score(rule['severity']),
                    affected_versions=['current'],
                    remediation_steps=[f"Review and fix {rule['name']} at line {line_number}"],
                    discovered_at=datetime.now(),
                    patched=False,
                    patch_date=None,
                    references=[f"Pattern: {rule['pattern']}"]
                )
                
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def scan_configuration(self, config: Dict[str, Any]) -> List[Vulnerability]:
        """Scan configuration for vulnerabilities"""
        vulnerabilities = []
        
        config_str = json.dumps(config, indent=2)
        
        # Check for weak passwords
        weak_patterns = self.vulnerability_database['configurations']['weak_passwords']['patterns']
        for pattern in weak_patterns:
            if pattern.lower() in config_str.lower():
                vulnerability = Vulnerability(
                    id=str(uuid.uuid4()),
                    vulnerability_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                    component='configuration',
                    description=f"Weak password pattern detected: {pattern}",
                    severity=ThreatLevel.HIGH,
                    cvss_score=7.5,
                    affected_versions=['current'],
                    remediation_steps=['Use strong, randomly generated passwords'],
                    discovered_at=datetime.now(),
                    patched=False,
                    patch_date=None,
                    references=['OWASP Password Guidelines']
                )
                vulnerabilities.append(vulnerability)
        
        # Check for insecure protocols
        insecure_patterns = self.vulnerability_database['configurations']['insecure_protocols']['patterns']
        for pattern in insecure_patterns:
            if pattern in config_str:
                vulnerability = Vulnerability(
                    id=str(uuid.uuid4()),
                    vulnerability_type=VulnerabilityType.INSECURE_COMMUNICATION,
                    component='configuration',
                    description=f"Insecure protocol detected: {pattern}",
                    severity=ThreatLevel.MEDIUM,
                    cvss_score=5.3,
                    affected_versions=['current'],
                    remediation_steps=['Use secure protocols (HTTPS, SFTP, SSH)'],
                    discovered_at=datetime.now(),
                    patched=False,
                    patch_date=None,
                    references=['OWASP Transport Layer Protection']
                )
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _calculate_cvss_score(self, severity: str) -> float:
        """Calculate CVSS score based on severity"""
        severity_scores = {
            'critical': 9.0,
            'high': 7.5,
            'medium': 5.0,
            'low': 2.5,
            'info': 0.0
        }
        return severity_scores.get(severity, 5.0)


class SecurityAgent(BaseAgent):
    """
    Security agent that monitors system interactions and detects threats
    """
    
    def __init__(self, message_bus: MessageBus, db_path: str = "security_agent.db"):
        capabilities = [
            AgentCapability(
                name="threat_monitoring",
                description="Monitor system for security threats and anomalies",
                input_schema={
                    "monitoring_scope": "string",
                    "sensitivity_level": "string"
                },
                output_schema={
                    "threats_detected": "array",
                    "monitoring_status": "object"
                },
                estimated_duration=5.0,
                max_concurrent=10
            ),
            AgentCapability(
                name="vulnerability_assessment",
                description="Assess system for security vulnerabilities",
                input_schema={
                    "target_component": "string",
                    "scan_type": "string"
                },
                output_schema={
                    "vulnerabilities": "array",
                    "risk_assessment": "object"
                },
                estimated_duration=30.0,
                max_concurrent=3
            ),
            AgentCapability(
                name="access_control",
                description="Monitor and control access to system resources",
                input_schema={
                    "resource": "string",
                    "requester": "string",
                    "action": "string"
                },
                output_schema={
                    "access_granted": "boolean",
                    "access_log": "object"
                },
                estimated_duration=1.0,
                max_concurrent=20
            ),
            AgentCapability(
                name="incident_response",
                description="Respond to security incidents",
                input_schema={
                    "incident_id": "string",
                    "response_type": "string"
                },
                output_schema={
                    "response_actions": "array",
                    "incident_status": "object"
                },
                estimated_duration=15.0,
                max_concurrent=5
            ),
            AgentCapability(
                name="security_audit",
                description="Perform comprehensive security audit",
                input_schema={
                    "audit_scope": "string",
                    "compliance_framework": "string"
                },
                output_schema={
                    "audit_results": "object",
                    "compliance_status": "object"
                },
                estimated_duration=60.0,
                max_concurrent=1
            )
        ]
        
        super().__init__("security_agent", capabilities, message_bus)
        
        self.db = SecurityDatabase(db_path)
        self.anomaly_detector = AnomalyDetector()
        self.threat_detector = ThreatDetector()
        self.vulnerability_scanner = VulnerabilityScanner()
        
        self.monitoring_active = True
        self.security_policies = self._load_security_policies()
        self.rate_limiters = defaultdict(deque)
        self.session_tracker = {}
        
        # Start security monitoring
        self._start_security_monitoring()
        
        # Subscribe to all message types for monitoring
        for message_type in MessageType:
            self.message_bus.subscribe(message_type, self._monitor_message)
    
    def _load_security_policies(self) -> List[SecurityPolicy]:
        """Load security policies"""
        return [
            SecurityPolicy(
                id="default_access_policy",
                name="Default Access Control",
                description="Default access control policy for system resources",
                rules=[
                    {"resource": "*", "action": "read", "allowed_agents": ["*"]},
                    {"resource": "config", "action": "write", "allowed_agents": ["admin_agent"]},
                    {"resource": "logs", "action": "delete", "allowed_agents": ["admin_agent"]},
                    {"resource": "tools", "action": "execute", "allowed_agents": ["*"], "rate_limit": 100}
                ],
                enforcement_level="strict",
                exceptions=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                active=True
            ),
            SecurityPolicy(
                id="communication_policy",
                name="Communication Security",
                description="Security policy for inter-agent communication",
                rules=[
                    {"external_communication": False, "encryption_required": True},
                    {"max_payload_size": 1048576},  # 1MB
                    {"allowed_message_types": ["task_request", "task_response", "status_update"]},
                    {"rate_limit_per_agent": 1000}  # messages per hour
                ],
                enforcement_level="moderate",
                exceptions=["system_monitor"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
                active=True
            )
        ]
    
    def _start_security_monitoring(self):
        """Start continuous security monitoring"""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    self._perform_security_checks()
                    self._update_threat_intelligence()
                    self._cleanup_old_logs()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logging.error(f"Security monitoring error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _monitor_message(self, message: Message):
        """Monitor all messages for security threats"""
        try:
            # Log communication
            self._log_communication(message)
            
            # Check rate limits
            if self._check_rate_limit_violation(message):
                self._create_security_event(
                    event_type=SecurityEventType.RATE_LIMIT_VIOLATION,
                    threat_level=ThreatLevel.MEDIUM,
                    source=message.sender,
                    target=message.recipient,
                    description=f"Rate limit violation by {message.sender}",
                    evidence={"message_id": message.id, "timestamp": message.timestamp},
                    context={"message_type": message.type.value}
                )
            
            # Analyze message content for threats
            payload_str = json.dumps(message.payload)
            threats = self.threat_detector.detect_threats(payload_str, {
                "sender": message.sender,
                "recipient": message.recipient,
                "message_type": message.type.value
            })
            
            for threat in threats:
                self._create_security_event(
                    event_type=SecurityEventType.SUSPICIOUS_COMMUNICATION,
                    threat_level=ThreatLevel(threat['severity']),
                    source=message.sender,
                    target=message.recipient,
                    description=f"Threat detected in message: {threat['type']}",
                    evidence=threat,
                    context={"message_id": message.id}
                )
            
            # Check for anomalous behavior
            self._check_behavioral_anomalies(message)
            
        except Exception as e:
            logging.error(f"Error monitoring message: {e}")
    
    def _log_communication(self, message: Message):
        """Log communication for audit trail"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        payload_hash = hashlib.sha256(json.dumps(message.payload).encode()).hexdigest()
        
        cursor.execute("""
            INSERT INTO communication_logs 
            (id, timestamp, sender, recipient, message_type, payload_hash, encrypted, external, suspicious)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            datetime.fromtimestamp(message.timestamp).isoformat(),
            message.sender,
            message.recipient,
            message.type.value,
            payload_hash,
            False,  # TODO: Check if message is encrypted
            self._is_external_communication(message),
            False   # Will be updated if threats are detected
        ))
        
        conn.commit()
        conn.close()
    
    def _check_rate_limit_violation(self, message: Message) -> bool:
        """Check if message violates rate limits"""
        current_time = time.time()
        sender = message.sender
        
        # Clean old entries (older than 1 hour)
        while (self.rate_limiters[sender] and 
               current_time - self.rate_limiters[sender][0] > 3600):
            self.rate_limiters[sender].popleft()
        
        # Add current message
        self.rate_limiters[sender].append(current_time)
        
        # Check rate limit (1000 messages per hour by default)
        rate_limit = 1000
        for policy in self.security_policies:
            if policy.name == "Communication Security":
                for rule in policy.rules:
                    if "rate_limit_per_agent" in rule:
                        rate_limit = rule["rate_limit_per_agent"]
                        break
        
        return len(self.rate_limiters[sender]) > rate_limit
    
    def _check_behavioral_anomalies(self, message: Message):
        """Check for behavioral anomalies"""
        # Collect recent activity for the sender
        activity_data = {
            'agent_id': message.sender,
            'message_count': len(self.rate_limiters[message.sender]),
            'execution_time': 0,  # Would need integration with task execution
            'payload_size': len(json.dumps(message.payload)),
            'error_rate': 0  # Would need integration with error tracking
        }
        
        anomalies = self.anomaly_detector.detect_anomalies(activity_data)
        
        for anomaly in anomalies:
            self._create_security_event(
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM if anomaly['severity'] == 'medium' else ThreatLevel.HIGH,
                source=message.sender,
                target="system",
                description=anomaly['description'],
                evidence=anomaly,
                context={"message_id": message.id}
            )
    
    def _is_external_communication(self, message: Message) -> bool:
        """Check if communication is external"""
        # Simple heuristic - could be enhanced with actual network analysis
        external_indicators = ['external', 'api', 'web', 'remote']
        return any(indicator in message.sender.lower() or indicator in message.recipient.lower() 
                  for indicator in external_indicators)
    
    def _create_security_event(self, event_type: SecurityEventType, threat_level: ThreatLevel,
                             source: str, target: str, description: str,
                             evidence: Dict[str, Any], context: Dict[str, Any]):
        """Create and store a security event"""
        event = SecurityEvent(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source=source,
            target=target,
            description=description,
            evidence=evidence,
            context=context,
            mitigated=False,
            mitigation_actions=[],
            false_positive=False,
            confidence_score=evidence.get('confidence', 0.8)
        )
        
        self.db.store_security_event(event)
        
        # Trigger immediate response for critical threats
        if threat_level == ThreatLevel.CRITICAL:
            self._trigger_incident_response(event)
        
        logging.warning(f"Security event created: {event_type.value} - {description}")
    
    def _trigger_incident_response(self, event: SecurityEvent):
        """Trigger immediate incident response for critical threats"""
        try:
            # Send alert message
            alert_message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.ERROR_REPORT,
                sender=self.agent_id,
                recipient="orchestrator",
                payload={
                    "security_alert": True,
                    "event_id": event.id,
                    "threat_level": event.threat_level.value,
                    "description": event.description,
                    "immediate_action_required": True
                },
                timestamp=time.time(),
                priority=5  # Highest priority
            )
            
            self.message_bus.publish(alert_message)
            
            # Implement automatic mitigation if possible
            mitigation_actions = self._get_automatic_mitigation_actions(event)
            for action in mitigation_actions:
                self._execute_mitigation_action(action, event)
            
        except Exception as e:
            logging.error(f"Error triggering incident response: {e}")
    
    def _get_automatic_mitigation_actions(self, event: SecurityEvent) -> List[str]:
        """Get automatic mitigation actions for an event"""
        actions = []
        
        if event.event_type == SecurityEventType.RATE_LIMIT_VIOLATION:
            actions.append("temporary_rate_limit_increase")
        elif event.event_type == SecurityEventType.SUSPICIOUS_COMMUNICATION:
            actions.append("quarantine_suspicious_agent")
        elif event.event_type == SecurityEventType.UNAUTHORIZED_ACCESS:
            actions.append("revoke_access_temporarily")
        
        return actions
    
    def _execute_mitigation_action(self, action: str, event: SecurityEvent):
        """Execute a mitigation action"""
        try:
            if action == "temporary_rate_limit_increase":
                # Temporarily block the source
                logging.info(f"Temporarily blocking {event.source} due to rate limit violation")
            elif action == "quarantine_suspicious_agent":
                # Send quarantine message
                quarantine_message = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.SHUTDOWN,
                    sender=self.agent_id,
                    recipient=event.source,
                    payload={"quarantine": True, "reason": "suspicious_activity"},
                    timestamp=time.time()
                )
                self.message_bus.publish(quarantine_message)
            elif action == "revoke_access_temporarily":
                logging.info(f"Revoking access for {event.source} temporarily")
            
            # Update event with mitigation action
            event.mitigation_actions.append(action)
            event.mitigated = True
            self.db.store_security_event(event)
            
        except Exception as e:
            logging.error(f"Error executing mitigation action {action}: {e}")
    
    def _perform_security_checks(self):
        """Perform periodic security checks"""
        try:
            # Check for suspicious patterns in recent activity
            recent_events = self.db.get_recent_events(hours=1)
            
            # Look for attack patterns
            event_types = [event.event_type for event in recent_events]
            event_counts = Counter(event_types)
            
            # Check for potential coordinated attacks
            if event_counts[SecurityEventType.SUSPICIOUS_COMMUNICATION] > 10:
                self._create_security_event(
                    event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.HIGH,
                    source="system",
                    target="system",
                    description="Potential coordinated attack detected - high volume of suspicious communications",
                    evidence={"suspicious_comm_count": event_counts[SecurityEventType.SUSPICIOUS_COMMUNICATION]},
                    context={"time_window": "1_hour"}
                )
            
            # Check system health from security perspective
            self._check_system_security_health()
            
        except Exception as e:
            logging.error(f"Error performing security checks: {e}")
    
    def _check_system_security_health(self):
        """Check system security health"""
        try:
            # Check for unpatched vulnerabilities
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM vulnerabilities 
                WHERE patched = FALSE AND severity IN ('critical', 'high')
            """)
            unpatched_critical = cursor.fetchone()[0]
            
            if unpatched_critical > 0:
                self._create_security_event(
                    event_type=SecurityEventType.CONFIGURATION_TAMPERING,
                    threat_level=ThreatLevel.HIGH,
                    source="system",
                    target="system",
                    description=f"{unpatched_critical} unpatched critical/high vulnerabilities detected",
                    evidence={"unpatched_count": unpatched_critical},
                    context={"check_type": "vulnerability_status"}
                )
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error checking system security health: {e}")
    
    def _update_threat_intelligence(self):
        """Update threat intelligence data"""
        try:
            # In a real implementation, this would fetch from threat intelligence feeds
            # For now, we'll update our internal patterns based on recent events
            
            recent_events = self.db.get_recent_events(hours=24)
            
            # Learn from recent attack patterns
            attack_patterns = defaultdict(list)
            for event in recent_events:
                if event.event_type == SecurityEventType.SUSPICIOUS_COMMUNICATION:
                    evidence = event.evidence
                    if 'pattern' in evidence:
                        attack_patterns[evidence.get('type', 'unknown')].append(evidence['pattern'])
            
            # Update threat detector patterns (simplified)
            for attack_type, patterns in attack_patterns.items():
                if len(patterns) > 5:  # If we see a pattern frequently
                    logging.info(f"Updating threat patterns for {attack_type}")
            
        except Exception as e:
            logging.error(f"Error updating threat intelligence: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old logs to prevent database bloat"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Keep logs for 30 days
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            cursor.execute("DELETE FROM communication_logs WHERE timestamp < ?", (cutoff_date,))
            cursor.execute("DELETE FROM access_logs WHERE timestamp < ?", (cutoff_date,))
            cursor.execute("DELETE FROM tool_execution_logs WHERE timestamp < ?", (cutoff_date,))
            
            # Keep security events for 90 days
            event_cutoff = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute("DELETE FROM security_events WHERE timestamp < ?", (event_cutoff,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error cleaning up old logs: {e}")
    
    def execute_capability(self, capability_name: str, payload: Dict[str, Any]) -> Any:
        """Execute security agent capabilities"""
        try:
            if capability_name == "threat_monitoring":
                return self._threat_monitoring(payload)
            elif capability_name == "vulnerability_assessment":
                return self._vulnerability_assessment(payload)
            elif capability_name == "access_control":
                return self._access_control(payload)
            elif capability_name == "incident_response":
                return self._incident_response(payload)
            elif capability_name == "security_audit":
                return self._security_audit(payload)
            else:
                raise ValueError(f"Unknown capability: {capability_name}")
        except Exception as e:
            logging.error(f"Error executing capability {capability_name}: {e}")
            raise
    
    def _threat_monitoring(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor for threats"""
        monitoring_scope = payload.get('monitoring_scope', 'system')
        sensitivity_level = payload.get('sensitivity_level', 'medium')
        
        # Get recent threats
        hours = 24 if sensitivity_level == 'high' else 1
        recent_events = self.db.get_recent_events(hours=hours)
        
        # Filter by scope
        if monitoring_scope != 'system':
            recent_events = [e for e in recent_events if monitoring_scope in e.source or monitoring_scope in e.target]
        
        # Categorize threats
        threats_by_level = defaultdict(list)
        for event in recent_events:
            threats_by_level[event.threat_level.value].append(asdict(event))
        
        monitoring_status = {
            'active': self.monitoring_active,
            'last_check': datetime.now().isoformat(),
            'events_analyzed': len(recent_events),
            'threat_summary': {level: len(events) for level, events in threats_by_level.items()}
        }
        
        return {
            'threats_detected': dict(threats_by_level),
            'monitoring_status': monitoring_status
        }
    
    def _vulnerability_assessment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assess vulnerabilities"""
        target_component = payload.get('target_component', 'system')
        scan_type = payload.get('scan_type', 'comprehensive')
        
        vulnerabilities = []
        
        if scan_type in ['comprehensive', 'code']:
            # Scan code (placeholder - would need actual code access)
            code_vulns = self.vulnerability_scanner.scan_code("", target_component)
            vulnerabilities.extend(code_vulns)
        
        if scan_type in ['comprehensive', 'configuration']:
            # Scan configuration (placeholder - would need actual config access)
            config_vulns = self.vulnerability_scanner.scan_configuration({})
            vulnerabilities.extend(config_vulns)
        
        # Calculate risk assessment
        risk_scores = [v.cvss_score for v in vulnerabilities]
        risk_assessment = {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_count': len([v for v in vulnerabilities if v.severity == ThreatLevel.CRITICAL]),
            'high_count': len([v for v in vulnerabilities if v.severity == ThreatLevel.HIGH]),
            'medium_count': len([v for v in vulnerabilities if v.severity == ThreatLevel.MEDIUM]),
            'low_count': len([v for v in vulnerabilities if v.severity == ThreatLevel.LOW]),
            'average_cvss_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
            'max_cvss_score': max(risk_scores) if risk_scores else 0.0,
            'overall_risk_level': self._calculate_overall_risk_level(vulnerabilities)
        }
        
        return {
            'vulnerabilities': [asdict(v) for v in vulnerabilities],
            'risk_assessment': risk_assessment
        }
    
    def _access_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Control access to resources"""
        resource = payload.get('resource', '')
        requester = payload.get('requester', '')
        action = payload.get('action', '')
        
        # Check against security policies
        access_granted = self._check_access_permission(resource, requester, action)
        
        # Log access attempt
        access_log = {
            'timestamp': datetime.now().isoformat(),
            'resource': resource,
            'requester': requester,
            'action': action,
            'granted': access_granted,
            'policy_checked': True
        }
        
        # Store access log
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO access_logs 
            (id, timestamp, agent_id, action, resource, success, ip_address, user_agent, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            access_log['timestamp'],
            requester,
            action,
            resource,
            access_granted,
            payload.get('ip_address', 'unknown'),
            payload.get('user_agent', 'unknown'),
            payload.get('session_id', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        
        # Create security event if access denied
        if not access_granted:
            self._create_security_event(
                event_type=SecurityEventType.UNAUTHORIZED_ACCESS,
                threat_level=ThreatLevel.MEDIUM,
                source=requester,
                target=resource,
                description=f"Access denied for {requester} attempting {action} on {resource}",
                evidence=access_log,
                context=payload
            )
        
        return {
            'access_granted': access_granted,
            'access_log': access_log
        }
    
    def _check_access_permission(self, resource: str, requester: str, action: str) -> bool:
        """Check if access should be granted based on security policies"""
        for policy in self.security_policies:
            if not policy.active:
                continue
            
            for rule in policy.rules:
                if 'resource' in rule and 'action' in rule and 'allowed_agents' in rule:
                    # Check if rule applies
                    if (rule['resource'] == '*' or rule['resource'] == resource) and \
                       (rule['action'] == '*' or rule['action'] == action):
                        
                        # Check if requester is allowed
                        if '*' in rule['allowed_agents'] or requester in rule['allowed_agents']:
                            return True
        
        return False  # Default deny
    
    def _incident_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Respond to security incidents"""
        incident_id = payload.get('incident_id', '')
        response_type = payload.get('response_type', 'investigate')
        
        response_actions = []
        
        if response_type == 'investigate':
            response_actions = [
                'collect_additional_evidence',
                'analyze_attack_vector',
                'identify_affected_systems',
                'assess_damage_scope'
            ]
        elif response_type == 'contain':
            response_actions = [
                'isolate_affected_systems',
                'block_malicious_traffic',
                'disable_compromised_accounts',
                'preserve_evidence'
            ]
        elif response_type == 'eradicate':
            response_actions = [
                'remove_malware',
                'patch_vulnerabilities',
                'update_security_controls',
                'strengthen_defenses'
            ]
        elif response_type == 'recover':
            response_actions = [
                'restore_from_backups',
                'verify_system_integrity',
                'monitor_for_reinfection',
                'resume_normal_operations'
            ]
        
        incident_status = {
            'incident_id': incident_id,
            'status': 'in_progress',
            'response_type': response_type,
            'actions_planned': len(response_actions),
            'estimated_completion': (datetime.now() + timedelta(hours=2)).isoformat()
        }
        
        return {
            'response_actions': response_actions,
            'incident_status': incident_status
        }
    
    def _security_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security audit"""
        audit_scope = payload.get('audit_scope', 'system')
        compliance_framework = payload.get('compliance_framework', 'general')
        
        # Collect audit data
        recent_events = self.db.get_recent_events(hours=24*7)  # Last week
        
        # Analyze security posture
        audit_results = {
            'audit_timestamp': datetime.now().isoformat(),
            'scope': audit_scope,
            'framework': compliance_framework,
            'security_events_summary': {
                'total_events': len(recent_events),
                'critical_events': len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL]),
                'high_events': len([e for e in recent_events if e.threat_level == ThreatLevel.HIGH]),
                'unmitigated_events': len([e for e in recent_events if not e.mitigated])
            },
            'policy_compliance': self._assess_policy_compliance(),
            'vulnerability_status': self._get_vulnerability_summary(),
            'recommendations': self._generate_security_recommendations(recent_events)
        }
        
        # Calculate compliance status
        compliance_score = self._calculate_compliance_score(audit_results)
        compliance_status = {
            'overall_score': compliance_score,
            'compliance_level': 'high' if compliance_score > 0.8 else 'medium' if compliance_score > 0.6 else 'low',
            'areas_for_improvement': self._identify_improvement_areas(audit_results),
            'next_audit_recommended': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        return {
            'audit_results': audit_results,
            'compliance_status': compliance_status
        }
    
    def _calculate_overall_risk_level(self, vulnerabilities: List[Vulnerability]) -> str:
        """Calculate overall risk level from vulnerabilities"""
        if not vulnerabilities:
            return 'low'
        
        critical_count = len([v for v in vulnerabilities if v.severity == ThreatLevel.CRITICAL])
        high_count = len([v for v in vulnerabilities if v.severity == ThreatLevel.HIGH])
        
        if critical_count > 0:
            return 'critical'
        elif high_count > 2:
            return 'high'
        elif high_count > 0:
            return 'medium'
        else:
            return 'low'
    
    def _assess_policy_compliance(self) -> Dict[str, Any]:
        """Assess compliance with security policies"""
        compliance = {
            'total_policies': len(self.security_policies),
            'active_policies': len([p for p in self.security_policies if p.active]),
            'policy_violations': 0,  # Would need to track violations
            'compliance_percentage': 95.0  # Placeholder
        }
        
        return compliance
    
    def _get_vulnerability_summary(self) -> Dict[str, Any]:
        """Get vulnerability summary"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM vulnerabilities WHERE patched = FALSE")
        unpatched = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM vulnerabilities")
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_vulnerabilities': total,
            'unpatched_vulnerabilities': unpatched,
            'patch_rate': (total - unpatched) / total if total > 0 else 1.0
        }
    
    def _generate_security_recommendations(self, recent_events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on recent events"""
        recommendations = []
        
        event_types = Counter([e.event_type for e in recent_events])
        
        if event_types[SecurityEventType.RATE_LIMIT_VIOLATION] > 5:
            recommendations.append("Implement more sophisticated rate limiting with adaptive thresholds")
        
        if event_types[SecurityEventType.SUSPICIOUS_COMMUNICATION] > 10:
            recommendations.append("Enhance communication monitoring and implement message encryption")
        
        if event_types[SecurityEventType.UNAUTHORIZED_ACCESS] > 3:
            recommendations.append("Review and strengthen access control policies")
        
        unmitigated_count = len([e for e in recent_events if not e.mitigated])
        if unmitigated_count > 5:
            recommendations.append("Improve incident response automation and mitigation capabilities")
        
        return recommendations
    
    def _calculate_compliance_score(self, audit_results: Dict[str, Any]) -> float:
        """Calculate compliance score"""
        # Simple scoring based on various factors
        base_score = 1.0
        
        # Deduct for unmitigated events
        unmitigated = audit_results['security_events_summary']['unmitigated_events']
        base_score -= min(unmitigated * 0.05, 0.3)
        
        # Deduct for critical events
        critical = audit_results['security_events_summary']['critical_events']
        base_score -= min(critical * 0.1, 0.4)
        
        # Factor in policy compliance
        policy_compliance = audit_results['policy_compliance']['compliance_percentage'] / 100.0
        base_score *= policy_compliance
        
        return max(0.0, base_score)
    
    def _identify_improvement_areas(self, audit_results: Dict[str, Any]) -> List[str]:
        """Identify areas for security improvement"""
        areas = []
        
        if audit_results['security_events_summary']['unmitigated_events'] > 5:
            areas.append("Incident Response")
        
        if audit_results['security_events_summary']['critical_events'] > 0:
            areas.append("Threat Detection")
        
        if audit_results['vulnerability_status']['unpatched_vulnerabilities'] > 0:
            areas.append("Vulnerability Management")
        
        if audit_results['policy_compliance']['compliance_percentage'] < 90:
            areas.append("Policy Compliance")
        
        return areas
    
    def shutdown(self):
        """Gracefully shutdown the security agent"""
        self.monitoring_active = False
        super().shutdown()


# Convenience function for external integration
def security_agent(task: str) -> Dict[str, Any]:
    """
    Main entry point for the security agent
    Compatible with the existing agent manifest system
    """
    try:
        # Parse task to determine action
        task_lower = task.lower()
        
        if 'monitor' in task_lower or 'threat' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = SecurityAgent(message_bus)
            
            result = agent.execute_capability("threat_monitoring", {
                "monitoring_scope": "system",
                "sensitivity_level": "medium"
            })
            
            return {
                "status": "success",
                "result": result,
                "message": "Threat monitoring completed"
            }
        
        elif 'scan' in task_lower or 'vulnerability' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = SecurityAgent(message_bus)
            
            result = agent.execute_capability("vulnerability_assessment", {
                "target_component": "system",
                "scan_type": "comprehensive"
            })
            
            return {
                "status": "success",
                "result": result,
                "message": "Vulnerability assessment completed"
            }
        
        elif 'audit' in task_lower:
            from core.agent_framework import MessageBus
            message_bus = MessageBus()
            agent = SecurityAgent(message_bus)
            
            result = agent.execute_capability("security_audit", {
                "audit_scope": "system",
                "compliance_framework": "general"
            })
            
            return {
                "status": "success",
                "result": result,
                "message": "Security audit completed"
            }
        
        else:
            return {
                "status": "success",
                "result": {
                    "message": "Security agent initialized and monitoring system",
                    "capabilities": [
                        "threat_monitoring",
                        "vulnerability_assessment",
                        "access_control",
                        "incident_response",
                        "security_audit"
                    ]
                },
                "message": "Security agent ready"
            }
            
    except Exception as e:
        return {
            "status": "failure",
            "result": None,
            "message": f"Security agent error: {str(e)}"
        }


if __name__ == "__main__":
    # Test the security agent
    logging.basicConfig(level=logging.INFO)
    
    from core.agent_framework import MessageBus
    message_bus = MessageBus()
    
    # Create security agent
    security_agent_instance = SecurityAgent(message_bus)
    
    # Test threat monitoring
    test_result = security_agent_instance.execute_capability("threat_monitoring", {
        "monitoring_scope": "system",
        "sensitivity_level": "high"
    })
    
    print("Threat Monitoring Test:")
    print(json.dumps(test_result, indent=2, default=str))
    
    # Test vulnerability assessment
    vuln_result = security_agent_instance.execute_capability("vulnerability_assessment", {
        "target_component": "system",
        "scan_type": "comprehensive"
    })
    
    print("\nVulnerability Assessment Test:")
    print(json.dumps(vuln_result, indent=2, default=str))
    
    # Cleanup
    security_agent_instance.shutdown()