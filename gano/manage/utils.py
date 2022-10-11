from pytorch_lightning.callbacks.progress import ProgressBar


class LightningProgressBar(ProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def update_args(default: dict, override: dict) -> dict:
    if default is None:
        return override

    for key, value in override.items():
        if key not in default:
            default[key] = value
        
    return default
