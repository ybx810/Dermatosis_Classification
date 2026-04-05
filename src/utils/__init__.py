from importlib import import_module

__all__ = [
    "build_image_level_metrics",
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "save_confusion_matrix_figure",
    "save_metrics_json",
    "export_training_visualizations",
    "plot_loss_curves",
    "plot_metric_curves",
]


def __getattr__(name: str):
    if name in {
        "build_image_level_metrics",
        "compute_classification_metrics",
        "compute_per_class_metrics",
        "save_confusion_matrix_figure",
        "save_metrics_json",
    }:
        module = import_module("src.utils.metrics")
        return getattr(module, name)
    if name in {"export_training_visualizations", "plot_loss_curves", "plot_metric_curves"}:
        module = import_module("src.utils.visualize")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")