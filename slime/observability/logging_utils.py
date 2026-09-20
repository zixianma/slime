import logging
import os
import uuid

import wandb

from slime.observability import wandb_utils
from slime.observability.tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False
_LOGGER_NAMESPACE = "slime-observability"
_LOGGER_ACTOR_HANDLE = None


class _WandbLoggerActor:
    """The sole W&B writer for a Ray training job.

    W&B's shared mode is not a reliable lifecycle primitive for checkpoint
    continuations: each worker can be interpreted as another resume owner.
    Serializing all metric writes through this actor preserves one run without
    requiring any worker process to attach to the cloud run.
    """

    def __init__(self, args):
        wandb_utils.init_wandb_primary(args)
        self.args = args
        self.log_count = 0

    def run_id(self):
        return self.args.wandb_run_id

    def ping(self):
        return True

    def log(self, metrics):
        wandb.log(metrics)
        self.log_count += 1

    def count(self):
        return self.log_count

    def finish(self):
        wandb_utils.wandb.finish()


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    global _LOGGER_ACTOR_HANDLE
    if not args.use_wandb:
        if primary:
            args.wandb_run_id = None
        return

    import ray

    if primary:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before the W&B logger actor")
        actor_name = f"slime-wandb-{os.environ.get('SLURM_JOB_ID', os.getpid())}-{uuid.uuid4().hex[:8]}"
        actor_cls = ray.remote(num_cpus=0)(_WandbLoggerActor)
        actor = actor_cls.options(name=actor_name, namespace=_LOGGER_NAMESPACE).remote(args)
        # Named actors are still reference-counted unless detached. Retain the
        # handle in the driver so the logger lives until explicit finish, while
        # still allowing Ray to clean it up automatically after a driver crash.
        _LOGGER_ACTOR_HANDLE = actor
        args.wandb_logger_actor_name = actor_name
        args.wandb_run_id = ray.get(actor.run_id.remote())
        return

    actor_name = getattr(args, "wandb_logger_actor_name", None)
    if not actor_name:
        raise RuntimeError("W&B worker is missing the single-writer logger actor identity")
    actor = ray.get_actor(actor_name, namespace=_LOGGER_NAMESPACE)
    if not ray.get(actor.ping.remote()):
        raise RuntimeError("W&B logger actor did not become ready")


def finish_tracking(args, primary: bool = True):
    global _LOGGER_ACTOR_HANDLE
    if not args.use_wandb:
        return
    if getattr(args, "wandb_logger_actor_name", None):
        if not primary:
            return
        import ray

        actor = _LOGGER_ACTOR_HANDLE or ray.get_actor(
            args.wandb_logger_actor_name, namespace=_LOGGER_NAMESPACE
        )
        ray.get(actor.finish.remote())
        ray.kill(actor, no_restart=True)
        _LOGGER_ACTOR_HANDLE = None
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        actor_name = getattr(args, "wandb_logger_actor_name", None)
        if actor_name:
            import ray

            actor = ray.get_actor(actor_name, namespace=_LOGGER_NAMESPACE)
            ray.get(actor.log.remote(metrics))
        else:
            wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
