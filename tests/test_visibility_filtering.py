# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

def test_execution_status_hook_filters_invisible_output_ports():
    from choregraph.hooks import ExecutionStatusHook
    from kedro.pipeline import node, Pipeline
    
    statuses = []
    def on_update(s):
        statuses.append(s)
        
    # Exclude "HiddenNode" (nodes whose only outputs are invisible)
    hook = ExecutionStatusHook(on_update=on_update, excluded_nodes={"HiddenNode"})
    
    n1 = node(lambda x: x, inputs="a", outputs="b", name="VisibleNode")
    n2 = node(lambda x: x, inputs="b", outputs="c", name="HiddenNode")
    p = Pipeline([n1, n2])
    
    hook.before_pipeline_run(run_params={}, pipeline=p, catalog=None)
    # Only VisibleNode should be in status
    assert "VisibleNode" in statuses[-1]
    assert "HiddenNode" not in statuses[-1]
    
    hook.before_node_run(node=n1, catalog=None, inputs={}, is_async=False, run_id="r1")
    assert statuses[-1]["VisibleNode"] == "running"
    
    # This should be ignored
    hook.before_node_run(node=n2, catalog=None, inputs={}, is_async=False, run_id="r1")
    assert "HiddenNode" not in statuses[-1]
    assert len(statuses[-1]) == 1
