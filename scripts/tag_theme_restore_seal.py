from mlflow.tracking import MlflowClient
c = MlflowClient("http://mlflow:5000")
rid = "0fc2986e0f49476694dfff3a4a955af5"
r = c.get_run(rid)
print("status", r.info.status)
if r.info.status == "RUNNING":
    c.set_terminated(rid, status="FINISHED")
c.set_tag(rid, "status", "FAILED")
c.set_tag(rid, "execution_state", "TRAIN_COMPLETE_FAILED")
c.set_tag(rid, "fail_gate", "radius_spread_gate")
c.set_tag(rid, "biology_gates_pass", "true")
c.set_tag(rid, "biology_pass", "false")  # card-level: need ALL gates
c.log_metric(rid, "seal_mean_theme_auprc", 0.6878633995850881)
c.log_metric(rid, "seal_n_themes_pass_floor", 5.0)
c.log_metric(rid, "seal_mean_spread", 0.13364381591478983)
c.log_metric(rid, "seal_mean_sat", 0.2602120190858841)
print("tagged", c.get_run(rid).info.status)
