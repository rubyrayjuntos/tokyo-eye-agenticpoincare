from mlflow.tracking import MlflowClient
c = MlflowClient("http://mlflow:5000")
rid = "d498d3b8d30049c9bcadc5bcef107e9c"
r = c.get_run(rid)
print("before", r.info.status)
if r.info.status == "RUNNING":
    c.set_terminated(rid, status="FAILED")
else:
    # already ended by crash cleanup — tag as aborted launch
    pass
c.set_tag(rid, "status", "LAUNCH_ABORTED_PERM")
c.set_tag(rid, "abort_reason", "PermissionError writing data/gates stamp")
print("after", c.get_run(rid).info.status)
