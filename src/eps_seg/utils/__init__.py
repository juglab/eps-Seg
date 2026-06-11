"""Utility package exports.

Schedule-report helpers used to live in ``eps_seg.utils.schedules``.  Some
entrypoints only need sibling modules such as ``eps_seg.utils.outputs``; keep
those imports working even when the optional schedules module is absent.
"""

try:
    from eps_seg.utils.schedules import (
        build_confidence_accuracy_table,
        build_disabled_row_table,
        build_evaluator_cohort_summary,
        build_new_row_class_balance,
        build_new_row_substack_origin,
        build_stage_summary_table,
        discover_schedule_paths,
        load_experiment_context,
        load_schedule,
        load_schedule_series,
        plot_confidence_vs_accuracy,
        plot_confidence_vs_accuracy_by_stage,
        plot_disabled_rows_by_origin,
        plot_new_row_class_balance,
        plot_new_row_substack_origin,
        schedule_to_frame,
        stage_new_rows,
    )
except ModuleNotFoundError as exc:
    if exc.name != "eps_seg.utils.schedules":
        raise
    __all__ = []
else:
    __all__ = [
        "build_confidence_accuracy_table",
        "build_disabled_row_table",
        "build_evaluator_cohort_summary",
        "build_new_row_class_balance",
        "build_new_row_substack_origin",
        "build_stage_summary_table",
        "discover_schedule_paths",
        "load_experiment_context",
        "load_schedule",
        "load_schedule_series",
        "plot_confidence_vs_accuracy",
        "plot_confidence_vs_accuracy_by_stage",
        "plot_disabled_rows_by_origin",
        "plot_new_row_class_balance",
        "plot_new_row_substack_origin",
        "schedule_to_frame",
        "stage_new_rows",
    ]
