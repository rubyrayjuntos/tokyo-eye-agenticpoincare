from mlflow.tracking import MlflowClient
c = MlflowClient("http://mlflow:5000")
rid = "0fc2986e0f49476694dfff3a4a955af5"
r = c.get_run(rid)
print("run_id", r.info.run_id)
print("status", r.info.status)
print("run_name", r.data.tags.get("mlflow.runName") or r.info.run_name)
print("--- tags ---")
for k,v in sorted(r.data.tags.items()):
    if k.startswith("mlflow."): continue
    print(f"{k}={v}")
print("--- metrics (filtered) ---")
for k,v in sorted(r.data.metrics.items()):
    if any(x in k for x in ("auprc","spread","sat","equiv","seal","theme","h_norm","probe")):
        print(f"{k}={v}")
