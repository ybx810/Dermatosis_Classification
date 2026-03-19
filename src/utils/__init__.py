from importlib import import_module

__all__ = [
    "aggregate_patch_predictions_to_image",
    "compute_classification_metrics",
    "compute_multilevel_classification_metrics",
    "resolve_evaluation_config",
    "save_confusion_matrix_figure",
    "save_metrics_json",
    "export_training_visualizations",
    "plot_loss_curves",
    "plot_metric_curves",
]


def __getattr__(name: str):
    if name in {
        "aggregate_patch_predictions_to_image",
        "compute_classification_metrics",
        "compute_multilevel_classification_metrics",
        "resolve_evaluation_config",
        "save_confusion_matrix_figure",
        "save_metrics_json",
    }:
        module = import_module("src.utils.metrics")
        return getattr(module, name)
    if name in {"export_training_visualizations", "plot_loss_curves", "plot_metric_curves"}:
        module = import_module("src.utils.visualize")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
