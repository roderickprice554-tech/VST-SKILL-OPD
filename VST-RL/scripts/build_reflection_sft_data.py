import argparse
import asyncio
import json
from pathlib import Path

from recurrent.reflection_sft import (
    OpenAICompatibleReflectionTeacher,
    build_teacher_request,
    generate_with_client,
    offline_record_to_trajectory,
    to_vst_sft_record,
    validate_teacher_response,
    write_jsonl_with_seeks,
)


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _summary(requests, accepted, rejected, source):
    return {
        "request_count": requests,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "fixture_count": accepted if source == "fixture" else 0,
        "external_count": accepted if source == "external" else 0,
    }


def export_requests(args):
    trajectories = [offline_record_to_trajectory(record) for record in _read_jsonl(args.trajectory_jsonl)]
    exported = []
    for trajectory in trajectories:
        request = build_teacher_request(trajectory)
        exported.append({"payload": request.payload, "analyzer_prompt": request.analyzer_prompt})
    write_jsonl_with_seeks(exported, args.requests_jsonl)
    print(json.dumps(_summary(len(trajectories), 0, 0, None), sort_keys=True))
    return 0


def import_responses(args):
    trajectories = [offline_record_to_trajectory(record) for record in _read_jsonl(args.trajectory_jsonl)]
    responses = {item["trajectory_uid"]: item["response"] for item in _read_jsonl(args.responses_jsonl)}
    records = []
    rejected = 0
    for trajectory in trajectories:
        try:
            accepted = validate_teacher_response(
                trajectory,
                responses[trajectory.trajectory_uid],
                source="fixture",
            )
        except (KeyError, ValueError):
            rejected += 1
            continue
        records.append(to_vst_sft_record(trajectory, accepted))
    if not records:
        raise ValueError("no accepted reflection SFT records")
    write_jsonl_with_seeks(records, args.output_jsonl)
    print(json.dumps(_summary(len(trajectories), len(records), rejected, "fixture"), sort_keys=True))
    return 0


async def call_teacher(args, *, client):
    trajectories = [offline_record_to_trajectory(record) for record in _read_jsonl(args.trajectory_jsonl)]
    records = []
    rejected = 0
    for trajectory in trajectories:
        try:
            accepted = await generate_with_client(trajectory, client)
        except ValueError:
            rejected += 1
            continue
        records.append(to_vst_sft_record(trajectory, accepted))
    if not records:
        raise ValueError("no accepted reflection SFT records")
    write_jsonl_with_seeks(records, args.output_jsonl)
    summary = _summary(len(trajectories), len(records), rejected, "external")
    print(json.dumps(summary, sort_keys=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--trajectory-jsonl", required=True)
    export.add_argument("--requests-jsonl", required=True)
    import_ = subparsers.add_parser("import")
    import_.add_argument("--trajectory-jsonl", required=True)
    import_.add_argument("--responses-jsonl", required=True)
    import_.add_argument("--output-jsonl", required=True)
    call = subparsers.add_parser("call")
    call.add_argument("--trajectory-jsonl", required=True)
    call.add_argument("--output-jsonl", required=True)
    call.add_argument("--model", required=True)
    call.add_argument("--api-key", required=True)
    call.add_argument("--base-url")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "export":
        return export_requests(args)
    if args.mode == "import":
        return import_responses(args)
    client = OpenAICompatibleReflectionTeacher(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    asyncio.run(call_teacher(args, client=client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
