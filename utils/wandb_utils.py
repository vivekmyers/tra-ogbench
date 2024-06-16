import tempfile

import absl.flags as flags
import ml_collections
import wandb


def get_flag_dict():
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
        entity=None,
        project='project',
        group=None,
        name=None,
        offline=False,
):
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode='offline' if offline else 'online',
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run
