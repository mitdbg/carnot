from carnot.execution.execution import Execution


def assert_planner_did_not_hit_max_steps(planner, result) -> None:
    """Assert planner did not hit max_steps limit."""
    from carnot.agents.utils import AgentMaxStepsError

    max_steps_message = "The agent did not return a final answer within the maximum number of steps."
    assert result != max_steps_message, (
        f"Planner hit max_steps limit ({planner.max_steps}). "
        "This simple task should complete in fewer steps."
    )
    if hasattr(planner, "planning_memory") and planner.planning_memory and planner.planning_memory.steps:
        last_step = planner.planning_memory.steps[-1]
        error = getattr(last_step, "error", None)
        assert not isinstance(error, AgentMaxStepsError), f"Planner hit max_steps: {error}"


def _plan_contains_sem_topk(logical_plan) -> bool:
    """Recursively check if plan contains SemanticTopK/sem_topk."""
    if isinstance(logical_plan, dict):
        params = logical_plan.get("params", {})
        if params.get("operator") == "SemanticTopK":
            return True
        for v in logical_plan.values():
            if _plan_contains_sem_topk(v):
                return True
    elif isinstance(logical_plan, list):
        for item in logical_plan:
            if _plan_contains_sem_topk(item):
                return True
    return False


def test_e2e_agent_uses_sem_topk_for_indexed_dataset(enron_dataset_with_hierarchical_index, llm_config):
    query = "Find emails about the Raptor investment or LJM partnerships"
    execution = Execution(
        query=query,
        datasets=[enron_dataset_with_hierarchical_index],
        llm_config=llm_config,
    )
    # Plan phase - agent should call data_discovery, see index, use sem_topk
    nl_plan, logical_plan = execution.plan()

    assert_planner_did_not_hit_max_steps(execution.planner, logical_plan)

    assert _plan_contains_sem_topk(logical_plan), (
        f"Expected plan to use sem_topk for indexed dataset. Plan: {logical_plan}"
    )

    execution._plan = logical_plan
    items, answer_str = execution.run()

    assert len(items) > 0, f"Expected results from sem_topk. Answer: {answer_str}"

    result_paths = [
        getattr(i, "path", i.get("path", "") if isinstance(i, dict) else "")
        for i in items
    ]
    raptor_keywords = ["kaminski", "delainey", "whalley", "parks", "raptor", "giron"]
    matches = sum(1 for p in result_paths for k in raptor_keywords if k in str(p).lower())
    assert matches >= 2, (
        f"Expected Raptor-related emails in results. Got {len(items)} items. "
        f"Sample paths: {result_paths[:5]}"
    )

    execution.planner.cleanup()
