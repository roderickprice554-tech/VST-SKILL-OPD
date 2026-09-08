#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "memory_transition_count",
    "final_turn_count",
    "trajectory_count",
    "reward_mapped",
    "correctness_mapped",
    "reflection_valid",
    "reflection_applied",
    "key_transition_count",
    "query_leakage_count",
    "top_k",
    "valid_token_count",
    "final_token_count",
    "final_opd_token_count",
    "teacher_detached",
    "rl_loss",
    "lopd_loss",
    "weighted_lopd_loss",
    "total_loss",
    "cache_bytes",
    "optimizer_completed",
    "trainable_param_max_change",
}


def audit_enabled_smoke(report: dict) -> None:
    missing = REQUIRED_FIELDS - set(report)
    if missing:
        raise ValueError(f"missing smoke fields: {sorted(missing)}")
    if report["memory_transition_count"] < 2:
        raise ValueError("smoke needs at least two memory transitions")
    if report["final_turn_count"] != report["trajectory_count"] or report["final_turn_count"] < 1:
        raise ValueError("each trajectory must have exactly one final turn")
    if report["reward_mapped"] is not True:
        raise ValueError("trajectory reward mapping was not verified")
    if report["correctness_mapped"] is not True:
        raise ValueError("trajectory correctness mapping was not verified")
    if report["reflection_valid"] < 1 or report["reflection_applied"] < 1:
        raise ValueError("smoke must contain a valid applied reflection")
    if report["key_transition_count"] < 1:
        raise ValueError("reflection must select at least one key transition")
    if report["query_leakage_count"] != 0:
        raise ValueError("query leakage was detected")
    if report["top_k"] != 100:
        raise ValueError("teacher cache must use top_k=100")
    if report["valid_token_count"] < 1:
        raise ValueError("global-episode OPD has no valid tokens")
    if report["final_token_count"] != 0:
        raise ValueError("final answer tokens entered OPD")
    if report["final_opd_token_count"] != 0:
        raise ValueError("final answer tokens entered global-episode OPD")
    if report["teacher_detached"] is not True:
        raise ValueError("teacher cache was not detached")
    for key in (
        "rl_loss",
        "lopd_loss",
        "weighted_lopd_loss",
        "total_loss",
        "trainable_param_max_change",
    ):
        if not math.isfinite(float(report[key])):
            raise ValueError(f"{key} is not finite")
    expected_total = float(report["rl_loss"]) + float(report["weighted_lopd_loss"])
    if not math.isclose(float(report["total_loss"]), expected_total, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("joint loss composition is inconsistent")
    if report["cache_bytes"] <= 0:
        raise ValueError("teacher cache byte count must be positive")
    if report["optimizer_completed"] is not True:
        raise ValueError("optimizer did not complete")
    if report["trainable_param_max_change"] <= 0:
        raise ValueError("no trainable Actor parameter changed")


def audit_disabled_smoke(report: dict) -> None:
    expected = {"skill_opd_enabled", "opd_branch_executed", "optimizer_completed", "rl_loss"}
    missing = expected - set(report)
    if missing:
        raise ValueError(f"missing disabled smoke fields: {sorted(missing)}")
    if report["skill_opd_enabled"] is not False:
        raise ValueError("disabled smoke unexpectedly enabled Skill OPD")
    if report["opd_branch_executed"] is not False:
        raise ValueError("disabled smoke executed the OPD branch")
    if report["optimizer_completed"] is not True:
        raise ValueError("disabled smoke optimizer did not complete")
    if not math.isfinite(float(report["rl_loss"])):
        raise ValueError("disabled smoke RL loss is not finite")


def audit_cpu_code_smoke(report: dict) -> None:
    audit_enabled_smoke(report)
    expected = {
        "smoke_type",
        "reflection_source",
        "teacher_source",
        "external_api_called",
        "device",
        "gpu_used",
        "episode_memory_row_count",
        "key_memory_row_count",
        "non_key_episode_row_count",
        "sft_target_token_count",
        "sft_loss",
        "sft_trainable_param_max_change",
        "actor_trainable_param_max_change",
        "disabled_path_equivalent",
        "retained_mass",
    }
    missing = expected - set(report)
    if missing:
        raise ValueError(f"missing CPU code smoke fields: {sorted(missing)}")
    if report["smoke_type"] != "cpu_code":
        raise ValueError("report is not labelled as a CPU code smoke")
    if report["reflection_source"] != "fixture":
        raise ValueError("CPU code smoke reflection source must be fixture")
    if report["teacher_source"] != "fixture":
        raise ValueError("CPU code smoke teacher source must be fixture")
    if report["external_api_called"] is not False:
        raise ValueError("CPU code smoke must not call an external API")
    if report["device"] != "cpu" or report["gpu_used"] is not False:
        raise ValueError("CPU code smoke must not claim GPU execution")
    if report["episode_memory_row_count"] != report["memory_transition_count"]:
        raise ValueError("episode skill must supervise every memory row")
    if report["key_memory_row_count"] < 1:
        raise ValueError("CPU code smoke needs a key memory row")
    if report["non_key_episode_row_count"] < 1:
        raise ValueError("CPU code smoke needs a non-key episode-only row")
    if report["sft_target_token_count"] < 1:
        raise ValueError("Reflection SFT has no assistant target tokens")
    for key in (
        "sft_loss",
        "sft_trainable_param_max_change",
        "actor_trainable_param_max_change",
    ):
        if not math.isfinite(float(report[key])):
            raise ValueError(f"{key} is not finite")
    if report["sft_trainable_param_max_change"] <= 0:
        raise ValueError("no Reflection SFT parameter changed")
    if report["actor_trainable_param_max_change"] <= 0:
        raise ValueError("no trainable Actor parameter changed")
    if report["disabled_path_equivalent"] is not True:
        raise ValueError("disabled path did not return the original RL loss")
    retained_mass = float(report["retained_mass"])
    if not math.isfinite(retained_mass) or not 0.0 < retained_mass <= 1.0:
        raise ValueError("teacher retained mass must be finite and in (0, 1]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--cpu-code", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.disabled and args.cpu_code:
        parser.error("--disabled and --cpu-code are mutually exclusive")
    if args.disabled:
        audit_disabled_smoke(report)
    elif args.cpu_code:
        audit_cpu_code_smoke(report)
    else:
        audit_enabled_smoke(report)
    print("Skill OPD smoke audit passed")


if __name__ == "__main__":
    main()
