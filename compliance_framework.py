"""
compliance_frameworks.py
------------------------
Defines control lists for CMMC, NIST SP 800-171, and SOC 2.
Each control has an ID, a short name, and a description that
the compliance agent uses to search for evidence in uploaded docs.
"""

FRAMEWORKS = {

    "CMMC": [
        {"id": "AC.1.001", "name": "Authorized Access Control",       "description": "Limit system access to authorized users, processes, and devices."},
        {"id": "AC.1.002", "name": "Limit Transaction Types",         "description": "Limit system access to the types of transactions and functions authorized users are permitted to execute."},
        {"id": "AC.2.005", "name": "User Notification",               "description": "Provide privacy and security notices consistent with CUI rules."},
        {"id": "AC.2.006", "name": "Least Privilege",                 "description": "Employ the principle of least privilege, including for specific security functions."},
        {"id": "IA.1.076", "name": "User Identification",             "description": "Identify system users, processes, and devices before allowing access."},
        {"id": "IA.1.077", "name": "User Authentication",             "description": "Authenticate the identities of users, processes, or devices before allowing access."},
        {"id": "IA.2.078", "name": "Password Complexity",             "description": "Enforce minimum password complexity and change requirements."},
        {"id": "IA.2.079", "name": "Password Prohibition",            "description": "Prohibit password reuse for a specified number of generations."},
        {"id": "AU.2.041", "name": "User Activity Audit",             "description": "Ensure that actions of individual users can be traced to those users."},
        {"id": "AU.2.042", "name": "Audit Log Review",                "description": "Review and update logged events."},
        {"id": "AU.3.045", "name": "Audit Correlation",               "description": "Review and analyze audit logs to identify unauthorized activity."},
        {"id": "CM.2.061", "name": "Baseline Configurations",         "description": "Establish and maintain baseline configurations for systems."},
        {"id": "CM.2.062", "name": "Security Configuration",          "description": "Establish and maintain security configuration settings."},
        {"id": "IR.2.092", "name": "Incident Response",               "description": "Establish an operational incident-handling capability."},
        {"id": "RM.2.141", "name": "Risk Assessment",                 "description": "Periodically assess the risk to operations from system use."},
        {"id": "SC.1.175", "name": "Communications Protection",       "description": "Monitor, control, and protect communications at external boundaries."},
        {"id": "SI.1.210", "name": "Malicious Code Protection",       "description": "Provide protection from malicious code at appropriate locations."},
        {"id": "SI.1.211", "name": "Security Alerts",                 "description": "Monitor system security alerts and take action in response."},
        {"id": "MP.2.120", "name": "Media Access Control",            "description": "Control access to CUI on system media."},
        {"id": "PE.1.131", "name": "Physical Access",                 "description": "Limit physical access to systems to authorized individuals."},
    ],

    "NIST": [
        {"id": "3.1.1",  "name": "Account Management",               "description": "Limit system access to authorized users and legitimate processes."},
        {"id": "3.1.2",  "name": "Authorized Transactions",          "description": "Limit access to types of transactions and functions authorized users may execute."},
        {"id": "3.1.3",  "name": "CUI Flow Control",                 "description": "Control the flow of CUI in accordance with approved authorizations."},
        {"id": "3.1.5",  "name": "Least Privilege",                  "description": "Employ the principle of least privilege, including for specific security functions."},
        {"id": "3.3.1",  "name": "Audit Logging",                    "description": "Create, protect, and retain system audit records to enable monitoring."},
        {"id": "3.3.2",  "name": "User Traceability",                "description": "Ensure that actions of individual system users can be uniquely traced."},
        {"id": "3.4.1",  "name": "Baseline Config",                  "description": "Establish and maintain baseline configurations of systems."},
        {"id": "3.4.2",  "name": "Config Change Control",            "description": "Establish and enforce security configuration settings."},
        {"id": "3.5.1",  "name": "User Identification",              "description": "Identify system users, processes, and devices."},
        {"id": "3.5.2",  "name": "User Authentication",              "description": "Authenticate identities of users, processes, and devices."},
        {"id": "3.5.3",  "name": "Multi-Factor Auth",                "description": "Use multi-factor authentication for local and network access."},
        {"id": "3.6.1",  "name": "Incident Response",                "description": "Establish an operational incident-handling capability for systems."},
        {"id": "3.6.2",  "name": "Incident Tracking",                "description": "Track, document, and report incidents to authorities."},
        {"id": "3.11.1", "name": "Risk Assessment",                  "description": "Periodically assess risk to operations from system use."},
        {"id": "3.11.2", "name": "Vulnerability Scan",               "description": "Scan for vulnerabilities in systems periodically."},
        {"id": "3.12.1", "name": "Security Assessment",              "description": "Periodically assess security controls to determine effectiveness."},
        {"id": "3.13.1", "name": "Boundary Protection",              "description": "Monitor, control, and protect communications at external boundaries."},
        {"id": "3.14.1", "name": "Malware Protection",               "description": "Identify, report, and correct information and system flaws."},
        {"id": "3.14.6", "name": "Security Monitoring",              "description": "Monitor systems to detect attacks and indicators of potential attacks."},
        {"id": "3.7.1",  "name": "Maintenance Policy",               "description": "Perform maintenance on systems and provide controls on tools used."},
    ],

    "SOC2": [
        {"id": "CC1.1", "name": "Control Environment",               "description": "The entity demonstrates a commitment to integrity and ethical values."},
        {"id": "CC1.2", "name": "Board Oversight",                   "description": "The board exercises oversight of the development of internal controls."},
        {"id": "CC2.1", "name": "Information Communication",         "description": "The entity obtains and uses relevant quality information."},
        {"id": "CC2.2", "name": "Internal Communication",            "description": "The entity communicates internally about objectives and responsibilities."},
        {"id": "CC2.3", "name": "External Communication",            "description": "The entity communicates with external parties regarding matters affecting controls."},
        {"id": "CC3.1", "name": "Risk Assessment Objectives",        "description": "The entity specifies objectives to identify and assess risks."},
        {"id": "CC3.2", "name": "Risk Identification",               "description": "The entity identifies risks to the achievement of its objectives."},
        {"id": "CC4.1", "name": "Control Monitoring",                "description": "The entity selects, develops, and performs monitoring activities."},
        {"id": "CC5.1", "name": "Control Selection",                 "description": "The entity selects and develops control activities that mitigate risks."},
        {"id": "CC6.1", "name": "Logical Access",                    "description": "The entity implements logical access security to protect assets."},
        {"id": "CC6.2", "name": "Access Credentials",                "description": "Prior to issuing credentials, the entity registers and authorizes users."},
        {"id": "CC6.3", "name": "Access Removal",                    "description": "The entity removes access when no longer required."},
        {"id": "CC6.6", "name": "Threat Protection",                 "description": "The entity implements controls to prevent or detect threats from outside."},
        {"id": "CC6.7", "name": "Data Transmission",                 "description": "The entity restricts transmission of data to authorized users."},
        {"id": "CC7.1", "name": "Vulnerability Management",          "description": "The entity uses detection and monitoring procedures to identify vulnerabilities."},
        {"id": "CC7.2", "name": "Anomaly Detection",                 "description": "The entity monitors infrastructure and software for anomalies."},
        {"id": "CC7.3", "name": "Incident Evaluation",               "description": "The entity evaluates security events to determine if they are incidents."},
        {"id": "CC8.1", "name": "Change Management",                 "description": "The entity authorizes, designs, and implements changes to infrastructure."},
        {"id": "CC9.1", "name": "Risk Mitigation",                   "description": "The entity identifies and assesses risks from business disruption."},
        {"id": "A1.1",  "name": "Availability Capacity",             "description": "The entity maintains and monitors capacity to meet its availability commitments."},
    ],
}