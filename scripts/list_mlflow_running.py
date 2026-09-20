from mlflow.tracking import MlflowClient
client = MlflowClient("http://mlflow:5000")
exp = client.get_experiment_by_name("tokyoeye/equiformer-v3-moe/geometric/hyperbolic-spine")
print("exp", exp.experiment_id if exp else None)
runs = client.search_runs([exp.experiment_id], max_results=20, order_by=["attributes.start_time DESC"])
for r in runs:
    print(r.info.status, r.info.run_id[:8], r.info.run_name or r.data.tags.get("mlflow.runName"), r.data.tags.get("status"), r.data.tags.get("card"))
