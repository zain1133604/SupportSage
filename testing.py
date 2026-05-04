from audit_db import AuditDatabase

db = AuditDatabase(db_path="A:/CODING FILES/audit_store.db")

# Show all ORDER_WRITE actions (address changes, cancels, modifies)
results = db.explain_control("ORDER_WRITE")

for event in results:
    print("---")
    print("Actor     :", event["actor"])
    print("Action    :", event["event_type"])
    print("Time      :", event["timestamp_utc"])
    print("Details   :", event["details_json"])
    print("Rollback  :", event["rollback_sql"])