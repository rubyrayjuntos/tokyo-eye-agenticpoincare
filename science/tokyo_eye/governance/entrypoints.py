"""MLflow Project entry points for Tokyo Eye governance.

MLflow path is self-contained: no filesystem seal restore fallbacks.
Legacy filesystem seals remain on disk untouched for archaeology.

**Ops:** Prefer ``--tracking-uri http://localhost:5000`` from the host
(UI + HTTP API). Inside Docker science containers use ``http://mlflow:5000``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from science.tokyo_eye.governance.evaluate import (
    PromoteBlocked,
    ThresholdSpec,
    assert_promotable,
)
from science.tokyo_eye.governance.import_weights import (
    ArtifactVerifyError,
    ImportBlocked,
    import_checkpoint_into_mlflow,
)
from science.tokyo_eye.governance.registry import (
    ALIAS_CHAMPION,
    ALIAS_EXPERIMENTAL,
    get_model_by_alias,
    get_model_version,
    register_run_model_version,
    set_model_alias,
)
from science.tokyo_eye.governance.resolve import ResolveError, resolve_alias_checkpoint
from science.tokyo_eye.governance.vault import (
    VaultError,
    VaultRequired,
    assert_vaulted,
    download_checkpoint,
    upload_checkpoint,
)
from science.tokyo_eye.governance.taxonomy import (
    experiment_path,
    mandatory_run_tags,
    validate_run_name,
)
from science.tokyo_eye.governance.thresholds import resolve_thresholds


def _cli_thresholds(args: argparse.Namespace) -> tuple[list[ThresholdSpec], dict]:
    """Resolve threshold pack: MLflow run/template first; local JSON is legacy."""
    return resolve_thresholds(
        thresholds_run_id=getattr(args, "thresholds_run_id", None),
        thresholds_path=getattr(args, "thresholds", None),
        tracking_uri=args.tracking_uri,
        domain=getattr(args, "domain", None),
        subsystem=getattr(args, "subsystem", None),
    )


def _add_threshold_args(p: argparse.ArgumentParser, *, required_any: bool = False) -> None:
    p.add_argument(
        "--thresholds-run-id",
        default=None,
        help="MLflow run that holds thresholds/thresholds.json (SSOT)",
    )
    p.add_argument(
        "--thresholds",
        default=None,
        required=False,
        help="Legacy local JSON {metric: min}; prefer --thresholds-run-id or template",
    )
    del required_any  # reserved for future argparse mutual-exclusion helpers


def cmd_ensure_experiment(args: argparse.Namespace) -> int:
    import mlflow
    from science.tokyo_eye.governance.registry import ensure_tracking

    ensure_tracking(args.tracking_uri)
    name = experiment_path(args.domain, args.subsystem)
    mlflow.set_experiment(name)
    print(json.dumps({"experiment": name}))
    return 0


def cmd_seed_thresholds(args: argparse.Namespace) -> int:
    """Log a threshold pack onto a tagged template run (does not touch aliases)."""
    from science.tokyo_eye.governance.thresholds import seed_threshold_pack_template

    pack = None
    if args.thresholds:
        pack = json.loads(Path(args.thresholds).read_text())
    elif args.pack_json:
        pack = json.loads(args.pack_json)
    out = seed_threshold_pack_template(
        domain=args.domain,
        subsystem=args.subsystem,
        pack=pack,
        tracking_uri=args.tracking_uri,
        package_revision=args.package_revision,
    )
    print(json.dumps(out))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    thresholds, provenance = _cli_thresholds(args)
    result = assert_promotable(
        run_id=args.run_id,
        thresholds=thresholds,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps({"passed": True, "checked": result.checked, "thresholds": provenance}))
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    """Register a model URI already inside MLflow (runs:/… or models:/…)."""
    if not args.model_uri:
        print(
            json.dumps(
                {
                    "error": "register requires --model-uri from MLflow "
                    "(use import-weights to copy a local file into MLflow first)"
                }
            ),
            file=sys.stderr,
        )
        return 2
    tags = {}
    if args.domain and args.subsystem and args.capability_goal:
        tags = mandatory_run_tags(
            domain=args.domain,
            subsystem=args.subsystem,
            capability_goal=validate_run_name(args.capability_goal),
            package_revision=args.package_revision,
        )
    out = register_run_model_version(
        model_uri=args.model_uri,
        run_id=args.run_id,
        tags=tags,
        tracking_uri=args.tracking_uri,
    )
    print(json.dumps(out))
    return 0


def _require_champion_vault(
    *,
    version: str,
    tracking_uri: str | None,
    confirm_champion: bool,
) -> None:
    if not confirm_champion:
        raise VaultRequired(
            "set-alias --alias champion requires --confirm-champion "
            "and a matching GitHub Release (vault-upload)."
        )
    meta = get_model_version(version=version, tracking_uri=tracking_uri)
    if meta is None:
        raise VaultRequired(f"TokyoEye version {version} not found")
    sha = str((meta.get("tags") or {}).get("checkpoint_sha256") or "")
    if not sha:
        raise VaultRequired(
            f"Version {version} has no checkpoint_sha256 tag. "
            "Run vault-upload then import-weights, or re-import with --confirm-champion."
        )
    assert_vaulted(sha256=sha, alias=ALIAS_CHAMPION)


def cmd_set_alias(args: argparse.Namespace) -> int:
    try:
        if args.alias == ALIAS_CHAMPION:
            _require_champion_vault(
                version=str(args.version),
                tracking_uri=args.tracking_uri,
                confirm_champion=bool(getattr(args, "confirm_champion", False)),
            )
        out = set_model_alias(
            version=args.version,
            alias=args.alias,
            tracking_uri=args.tracking_uri,
        )
    except VaultError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(out))
    return 0


def cmd_import_weights(args: argparse.Namespace) -> int:
    """Copy checkpoint bytes into MLflow; never deletes the source file."""
    metrics = None
    if args.metrics:
        metrics = {k: float(v) for k, v in json.loads(Path(args.metrics).read_text()).items()}
    thresholds = None
    if args.thresholds_run_id or args.thresholds or (args.domain and args.subsystem):
        try:
            thresholds, _ = _cli_thresholds(args)
        except ValueError:
            if args.thresholds_run_id or args.thresholds:
                raise
    try:
        out = import_checkpoint_into_mlflow(
            checkpoint_path=args.checkpoint,
            domain=args.domain,
            subsystem=args.subsystem,
            capability_goal=args.capability_goal,
            tracking_uri=args.tracking_uri,
            alias=args.alias,
            metrics=metrics,
            thresholds=thresholds,
            package_revision=args.package_revision,
            overwrite_alias=bool(args.overwrite_alias),
            log_pyfunc=not args.no_pyfunc,
            evidence_artifact=args.evidence,
            confirm_champion=bool(getattr(args, "confirm_champion", False)),
        )
    except ImportBlocked as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    except PromoteBlocked as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    except (VaultError, ArtifactVerifyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **out}))
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    """Resolve models:/TokyoEye@alias from MLflow only (no HEALTHY fallback)."""
    try:
        out = resolve_alias_checkpoint(
            alias=args.alias,
            tracking_uri=args.tracking_uri,
        )
    except ResolveError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **out}))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Open a taxonomy experiment run; optionally exec or run full pipeline.

    Does not set @champion. Use --pipeline to chain evaluate→register→@experimental
    after training (or after --skip-train with a checkpoint).
    """
    if args.pipeline:
        from science.tokyo_eye.governance.train_pipeline import (
            run_train_evaluate_register_experimental,
        )

        metrics = None
        if args.metrics:
            metrics = {
                k: float(v) for k, v in json.loads(Path(args.metrics).read_text()).items()
            }
        thr = None
        thr_provenance = None
        try:
            thr_list, thr_provenance = resolve_thresholds(
                thresholds_run_id=args.thresholds_run_id,
                thresholds_path=args.thresholds,
                tracking_uri=args.tracking_uri,
                domain=args.domain,
                subsystem=args.subsystem,
            )
            thr = {s.metric: s.minimum for s in thr_list}
        except ValueError:
            if args.thresholds_run_id or args.thresholds:
                raise
            thr = None
            thr_provenance = None

        def _train_callable(run_id: str) -> Path:
            del run_id
            # Delegate to real trainer joined to the active governance run.
            import importlib
            import sys

            trainer = importlib.import_module(
                "experiments.training.v8.run_v8_experiment"
            )
            argv_backup = sys.argv
            try:
                sys.argv = [
                    "run_v8_experiment",
                    "--smoke",
                    "--join-active-mlflow-run",
                    "--taxonomy-domain",
                    args.domain,
                    "--taxonomy-subsystem",
                    args.subsystem,
                    "--run-name",
                    args.capability_goal,
                    "--device",
                    args.device or "cpu",
                    "--mlflow-uri",
                    args.tracking_uri or "http://mlflow:5000",
                ]
                if args.epochs:
                    sys.argv.extend(["--epochs", str(args.epochs)])
                trainer.main()
            finally:
                sys.argv = argv_backup
            # Best ckpt path convention from trainer
            out = Path("checkpoints/tokyoeye/runs") / args.capability_goal / "tokyoeye_best.pt"
            if not out.is_file():
                # fallback: newest tokyoeye_best under runs
                cands = sorted(
                    Path("checkpoints/tokyoeye/runs").glob("**/tokyoeye_best.pt"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if not cands:
                    raise FileNotFoundError("trainer did not write tokyoeye_best.pt")
                out = cands[0]
            return out

        out = run_train_evaluate_register_experimental(
            domain=args.domain,
            subsystem=args.subsystem,
            capability_goal=args.capability_goal,
            tracking_uri=args.tracking_uri,
            package_revision=args.package_revision,
            thresholds=thr,
            train_callable=None if args.skip_train else _train_callable,
            best_checkpoint=args.checkpoint,
            skip_train=bool(args.skip_train),
            set_experimental=not bool(args.no_experimental_alias),
            metrics=metrics,
        )
        if thr_provenance:
            out["thresholds_provenance"] = thr_provenance
        print(json.dumps(out))
        return 0

    import mlflow
    from science.tokyo_eye.governance.registry import ensure_tracking
    from science.tokyo_eye.governance.train_pipeline import ensure_taxonomy_experiment

    ensure_tracking(args.tracking_uri)
    exp = ensure_taxonomy_experiment(
        args.domain, args.subsystem, tracking_uri=args.tracking_uri
    )
    goal = validate_run_name(args.capability_goal)
    tags = mandatory_run_tags(
        domain=args.domain,
        subsystem=args.subsystem,
        capability_goal=goal,
        package_revision=args.package_revision,
    )
    mlflow.set_experiment(exp)
    with mlflow.start_run(run_name=goal) as run:
        for k, v in tags.items():
            mlflow.set_tag(k, v)
        mlflow.set_tag("governance_train_shell", "true")
        payload = {
            "experiment": exp,
            "run_id": run.info.run_id,
            "tags": tags,
            "note": (
                "MLflow-native train shell. Use --pipeline for "
                "evaluate→register→@experimental. @champion is never set here."
            ),
        }
        if args.exec:
            env = os.environ.copy()
            if args.tracking_uri:
                env["MLFLOW_TRACKING_URI"] = args.tracking_uri
            env["MLFLOW_EXPERIMENT_NAME"] = exp
            env["TOKYOEYE_GOVERNANCE_RUN_ID"] = run.info.run_id
            env["TOKYOEYE_DOMAIN"] = tags["domain"]
            env["TOKYOEYE_SUBSYSTEM"] = tags["subsystem"]
            env["TOKYOEYE_CAPABILITY_GOAL"] = tags["capability_goal"]
            cmd = shlex.split(args.exec)
            proc = subprocess.run(cmd, env=env, check=False)
            payload["exec"] = cmd
            payload["exec_returncode"] = proc.returncode
            mlflow.log_dict(payload, "governance_train.json")
            print(json.dumps(payload))
            return int(proc.returncode)
        mlflow.log_dict(payload, "governance_train.json")
        print(json.dumps(payload))
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """evaluate → import/register → set_alias (hard order)."""
    thresholds, provenance = _cli_thresholds(args)
    try:
        assert_promotable(
            run_id=args.run_id,
            thresholds=thresholds,
            tracking_uri=args.tracking_uri,
        )
    except PromoteBlocked as exc:
        print(
            json.dumps(
                {"passed": False, "error": str(exc), "thresholds": provenance}
            ),
            file=sys.stderr,
        )
        return 2

    if args.checkpoint:
        # Promote path that still needs local bytes: copy into MLflow first.
        metrics = None
        if args.metrics:
            metrics = {
                k: float(v) for k, v in json.loads(Path(args.metrics).read_text()).items()
            }
        try:
            out = import_checkpoint_into_mlflow(
                checkpoint_path=args.checkpoint,
                domain=args.domain,
                subsystem=args.subsystem,
                capability_goal=args.capability_goal,
                tracking_uri=args.tracking_uri,
                alias=args.alias,
                metrics=metrics,
                thresholds=thresholds,
                package_revision=args.package_revision,
                overwrite_alias=True,
                log_pyfunc=not args.no_pyfunc,
                confirm_champion=bool(getattr(args, "confirm_champion", False)),
            )
        except (VaultError, ArtifactVerifyError) as exc:
            print(
                json.dumps({"passed": False, "error": str(exc), "thresholds": provenance}),
                file=sys.stderr,
            )
            return 2
        print(json.dumps({"passed": True, "thresholds": provenance, **out}))
        return 0

    if not args.model_uri:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "provide --model-uri (MLflow) or --checkpoint (import into MLflow)",
                }
            ),
            file=sys.stderr,
        )
        return 2

    tags = mandatory_run_tags(
        domain=args.domain,
        subsystem=args.subsystem,
        capability_goal=validate_run_name(args.capability_goal),
        package_revision=args.package_revision,
    )
    reg = register_run_model_version(
        model_uri=args.model_uri,
        run_id=args.run_id,
        tags=tags,
        tracking_uri=args.tracking_uri,
    )
    try:
        if args.alias == ALIAS_CHAMPION:
            _require_champion_vault(
                version=reg["version"],
                tracking_uri=args.tracking_uri,
                confirm_champion=bool(getattr(args, "confirm_champion", False)),
            )
        alias = set_model_alias(
            version=reg["version"],
            alias=args.alias,
            tracking_uri=args.tracking_uri,
        )
    except VaultError as exc:
        print(
            json.dumps({"passed": False, "error": str(exc), "thresholds": provenance}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"register": reg, "alias": alias, "passed": True, "thresholds": provenance}))
    return 0


def cmd_vault_upload(args: argparse.Namespace) -> int:
    try:
        out = upload_checkpoint(
            checkpoint_path=args.checkpoint,
            alias=args.alias,
            capability_goal=args.capability_goal,
            repo=args.vault_repo,
        )
    except VaultError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **out.as_dict()}))
    return 0


def cmd_vault_restore(args: argparse.Namespace) -> int:
    dest = Path(args.dest or "checkpoints/tokyoeye/vault")
    try:
        if args.tag:
            path = download_checkpoint(
                tag=args.tag,
                dest_dir=dest,
                expected_sha256=args.expected_sha256,
                repo=args.vault_repo,
            )
            print(json.dumps({"ok": True, "path": str(path), "tag": args.tag}))
            return 0
        meta = get_model_by_alias(alias=args.alias, tracking_uri=args.tracking_uri)
        if meta is None:
            raise VaultRequired(f"No model version for @{args.alias}")
        tags = dict(meta.get("tags") or {})
        tag = args.tag or tags.get("vault_release_tag") or tags.get("vault_moving_tag")
        if not tag:
            raise VaultRequired(
                f"@{args.alias} has no vault_release_tag. Upload a Release first."
            )
        path = download_checkpoint(
            tag=tag,
            dest_dir=dest,
            expected_sha256=args.expected_sha256 or tags.get("checkpoint_sha256"),
            repo=args.vault_repo or tags.get("vault_repo"),
        )
    except VaultError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "path": str(path), "tag": tag, "alias": args.alias}))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    meta = get_model_by_alias(alias=args.alias, tracking_uri=args.tracking_uri)
    if meta is None:
        print(
            json.dumps({"ok": False, "error": f"No model version for @{args.alias}"}),
            file=sys.stderr,
        )
        return 2
    tags = dict(meta.get("tags") or {})
    sha = tags.get("checkpoint_sha256")
    mlflow_ok = False
    mlflow_error = None
    try:
        resolved = resolve_alias_checkpoint(
            alias=args.alias,
            tracking_uri=args.tracking_uri,
        )
        mlflow_ok = True
        if sha and resolved.get("sha256") and resolved["sha256"] != sha:
            mlflow_ok = False
            mlflow_error = "MLflow cache sha256 does not match version tag"
    except ResolveError as exc:
        mlflow_error = str(exc)
        resolved = None
    vault_ok = False
    vault_error = None
    if sha:
        try:
            vault = assert_vaulted(sha256=sha, alias=args.alias, repo=args.vault_repo)
            vault_ok = True
        except VaultError as exc:
            vault_error = str(exc)
            vault = None
    else:
        vault = None
        vault_error = "version has no checkpoint_sha256 tag"
    ok = vault_ok if args.alias == ALIAS_CHAMPION else mlflow_ok
    payload = {
        "ok": ok,
        "alias": args.alias,
        "version": meta.get("version"),
        "mlflow_ok": mlflow_ok,
        "mlflow_error": mlflow_error,
        "vault_ok": vault_ok,
        "vault_error": vault_error,
        "resolved": resolved,
        "vault": vault.as_dict() if vault else None,
    }
    if not ok:
        print(json.dumps(payload), file=sys.stderr)
        return 2
    print(json.dumps(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tokyoeye-governance")
    p.add_argument("--tracking-uri", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("ensure-experiment")
    e.add_argument("--domain", required=True)
    e.add_argument("--subsystem", required=True)
    e.set_defaults(func=cmd_ensure_experiment)

    st = sub.add_parser(
        "seed-thresholds",
        help="Log thresholds/thresholds.json onto a tagged template run (SSOT)",
    )
    st.add_argument("--domain", required=True)
    st.add_argument("--subsystem", required=True)
    st.add_argument("--thresholds", default=None, help="Optional local JSON to seed from")
    st.add_argument("--pack-json", default=None, help="Inline JSON object string")
    st.add_argument("--package-revision", default=None)
    st.set_defaults(func=cmd_seed_thresholds)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--run-id", required=True)
    ev.add_argument("--domain", default=None, help="For template-run threshold lookup")
    ev.add_argument("--subsystem", default=None, help="For template-run threshold lookup")
    _add_threshold_args(ev)
    ev.set_defaults(func=cmd_evaluate)

    r = sub.add_parser("register", help="Register an existing MLflow model URI")
    r.add_argument("--run-id", default=None)
    r.add_argument("--model-uri", default=None)
    r.add_argument("--domain", default=None)
    r.add_argument("--subsystem", default=None)
    r.add_argument("--capability-goal", default=None)
    r.add_argument("--package-revision", default=None)
    r.set_defaults(func=cmd_register)

    a = sub.add_parser("set-alias")
    a.add_argument("--version", required=True)
    a.add_argument(
        "--alias",
        required=True,
        choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL],
    )
    a.add_argument(
        "--confirm-champion",
        action="store_true",
        help="Required to move @champion; also requires a matching GitHub Release",
    )
    a.set_defaults(func=cmd_set_alias)

    im = sub.add_parser(
        "import-weights",
        help="Copy a local checkpoint into MLflow (source file is not deleted)",
    )
    im.add_argument("--checkpoint", required=True)
    im.add_argument("--domain", required=True)
    im.add_argument("--subsystem", required=True)
    im.add_argument("--capability-goal", required=True)
    im.add_argument(
        "--alias",
        default=None,
        choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL, "none", "null", "-"],
    )
    im.add_argument("--metrics", default=None, help="JSON metrics to log")
    _add_threshold_args(im)
    im.add_argument("--package-revision", default=None)
    im.add_argument("--evidence", default=None, help="Optional evidence JSON to log")
    im.add_argument("--overwrite-alias", action="store_true")
    im.add_argument("--no-pyfunc", action="store_true")
    im.add_argument(
        "--confirm-champion",
        action="store_true",
        help="Required with --alias champion; uploads GitHub Release then aliases",
    )
    im.set_defaults(func=cmd_import_weights)

    rs = sub.add_parser("resolve", help="Download models:/TokyoEye@alias (MLflow only)")
    rs.add_argument("--alias", default=ALIAS_CHAMPION, choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL])
    rs.set_defaults(func=cmd_resolve)

    tr = sub.add_parser("train", help="MLflow taxonomy train shell / pipeline (no Make)")
    tr.add_argument("--domain", required=True)
    tr.add_argument("--subsystem", required=True)
    tr.add_argument("--capability-goal", required=True)
    tr.add_argument("--package-revision", default=None)
    tr.add_argument("--exec", default=None, help="Optional shell command; inherits MLflow env")
    tr.add_argument(
        "--pipeline",
        action="store_true",
        help="Chain train→evaluate→register→@experimental (@champion never set)",
    )
    tr.add_argument("--skip-train", action="store_true")
    tr.add_argument("--checkpoint", default=None)
    _add_threshold_args(tr)
    tr.add_argument("--metrics", default=None)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--epochs", default=None)
    tr.add_argument(
        "--no-experimental-alias",
        action="store_true",
        help="Register but do not move @experimental",
    )
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("promote")
    pr.add_argument("--run-id", required=True)
    pr.add_argument("--domain", required=True)
    pr.add_argument("--subsystem", required=True)
    pr.add_argument("--capability-goal", required=True)
    pr.add_argument("--alias", required=True, choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL])
    pr.add_argument("--model-uri", default=None)
    pr.add_argument("--checkpoint", default=None)
    pr.add_argument("--metrics", default=None)
    pr.add_argument("--package-revision", default=None)
    pr.add_argument("--no-pyfunc", action="store_true")
    pr.add_argument(
        "--confirm-champion",
        action="store_true",
        help="Required to promote @champion; evaluate + Release + checksum",
    )
    _add_threshold_args(pr)
    pr.set_defaults(func=cmd_promote)

    vu = sub.add_parser("vault-upload", help="Copy a checkpoint into a GitHub Release")
    vu.add_argument("--checkpoint", required=True)
    vu.add_argument("--alias", required=True, choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL])
    vu.add_argument("--capability-goal", required=True)
    vu.add_argument("--vault-repo", default=None)
    vu.set_defaults(func=cmd_vault_upload)

    vr = sub.add_parser("vault-restore", help="Download a GitHub Release checkpoint")
    vr.add_argument("--alias", default=ALIAS_CHAMPION, choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL])
    vr.add_argument("--tag", default=None, help="Immutable or moving release tag")
    vr.add_argument("--dest", default=None)
    vr.add_argument("--expected-sha256", default=None)
    vr.add_argument("--vault-repo", default=None)
    vr.set_defaults(func=cmd_vault_restore)

    vf = sub.add_parser("verify", help="Check MLflow artifacts and GitHub Release checksums")
    vf.add_argument("--alias", default=ALIAS_CHAMPION, choices=[ALIAS_CHAMPION, ALIAS_EXPERIMENTAL])
    vf.add_argument("--vault-repo", default=None)
    vf.set_defaults(func=cmd_verify)

    return p


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "null", "-"}:
        return None
    return cleaned


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.tracking_uri = _empty_to_none(getattr(args, "tracking_uri", None))
    for key in (
        "run_id",
        "model_uri",
        "checkpoint",
        "domain",
        "subsystem",
        "capability_goal",
        "package_revision",
        "metrics",
        "thresholds",
        "thresholds_run_id",
        "pack_json",
        "evidence",
        "exec",
        "alias",
        "device",
        "epochs",
        "vault_repo",
        "dest",
        "tag",
        "expected_sha256",
    ):
        if hasattr(args, key):
            setattr(args, key, _empty_to_none(getattr(args, key)))
    if args.command == "register" and not args.model_uri:
        parser.error("register requires --model-uri (use import-weights for local files)")
    if args.command == "promote" and not args.model_uri and not args.checkpoint:
        parser.error("promote requires --model-uri or --checkpoint")
    if args.command in {"evaluate", "promote"}:
        has_thr = bool(getattr(args, "thresholds_run_id", None) or getattr(args, "thresholds", None))
        has_tmpl = bool(getattr(args, "domain", None) and getattr(args, "subsystem", None))
        if not has_thr and not has_tmpl:
            parser.error(
                f"{args.command} requires --thresholds-run-id, --thresholds, "
                "or --domain + --subsystem (template lookup)"
            )
    if args.command == "import-weights" and not args.checkpoint:
        parser.error("import-weights requires --checkpoint")
    if args.command == "train" and args.pipeline and args.skip_train and not args.checkpoint:
        parser.error("train --pipeline --skip-train requires --checkpoint")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
