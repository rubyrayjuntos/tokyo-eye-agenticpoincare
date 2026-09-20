import mlflow
from mlflow.tracking import MlflowClient

RUN_ID = "b0480b85e87f492f965f2993a9298dba"
mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()
run = client.get_run(RUN_ID)
print("before", run.info.run_id, run.info.status, run.info.end_time)

client.set_terminated(RUN_ID, status="FINISHED")
client.set_tag(RUN_ID, "status", "QUALIFIED")
client.set_tag(RUN_ID, "execution_state", "TRAIN_COMPLETE_QUALIFIED")
client.set_tag(RUN_ID, "frontend", "equiformer_v3_pool")
client.set_tag(RUN_ID, "lift", "geoopt.PoincareBall")
client.set_tag(RUN_ID, "forbid_se3_lite", "true")
client.set_tag(RUN_ID, "biology_pass", "false")
client.log_metric(RUN_ID, "seal_mean_sat", 0.2940177917480469)
client.log_metric(RUN_ID, "seal_mean_spread", 0.2313267116745313)
client.log_metric(RUN_ID, "seal_mean_h_norm", 0.9943032924163183)
client.log_metric(RUN_ID, "seal_equiv_lift", 5.267121e-08)
client.log_metric(RUN_ID, "seal_equiv_full", 9.548387e-07)
client.log_metric(RUN_ID, "seal_qualified", 1.0)

run2 = client.get_run(RUN_ID)
print("after", run2.info.run_id, run2.info.status, run2.info.end_time)
print("tags.status", run2.data.tags.get("status"), run2.data.tags.get("execution_state"))
