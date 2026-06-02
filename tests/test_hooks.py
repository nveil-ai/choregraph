# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations


from kedro.pipeline import node, Pipeline

from choregraph.hooks import ExecutionStatusHook


# def test_viz_events_hook_writes_events_file(tmp_path: Path):
#     hook = VizEventsHook(project_path=tmp_path)

#     n = node(lambda x: x, inputs="a", outputs="b", name="MyNode")
#     p = Pipeline([n])

#     hook.before_pipeline_run(run_params={"run_id": "r1"}, pipeline=p, catalog=None)
#     hook.before_node_run(node=n, catalog=None, inputs={}, is_async=False, run_id="r1")
#     hook.after_node_run(node=n, catalog=None, inputs={}, outputs={}, is_async=False, run_id="r1")
#     hook.after_pipeline_run(run_params={"run_id": "r1"}, run_result=None, pipeline=p, catalog=None)

#     events_path = tmp_path / ".viz" / "kedro_pipeline_events.json"
#     assert events_path.exists()

#     data = json.loads(events_path.read_text(encoding="utf-8"))
#     assert data[0]["event_type"] == "before_pipeline_run"
#     assert data[-1]["event_type"] == "after_pipeline_run"


def test_execution_status_hook_tracks_status_changes():
    statuses = []

    def on_update(s):
        statuses.append(s)

    hook = ExecutionStatusHook(on_update=on_update)

    n1 = node(lambda x: x, inputs="a", outputs="b", name="Node1")
    n2 = node(lambda x: x, inputs="b", outputs="c", name="Node2")
    p = Pipeline([n1, n2])

    hook.before_pipeline_run(run_params={}, pipeline=p, catalog=None)
    assert statuses[-1] == {"Node1": "pending", "Node2": "pending"}

    hook.before_node_run(node=n1, catalog=None, inputs={}, is_async=False, run_id="r1")
    assert statuses[-1]["Node1"] == "running"

    hook.after_node_run(node=n1, catalog=None, inputs={}, outputs={}, is_async=False, run_id="r1")
    assert statuses[-1]["Node1"] == "completed"

    hook.on_node_error(error=RuntimeError("boom"), node=n2, catalog=None, inputs={}, is_async=False, run_id="r1")
    assert statuses[-1]["Node2"] == "failed"
