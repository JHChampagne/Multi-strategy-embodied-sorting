"""Unified execution entrypoint with an explicit M1 compatibility adapter."""


class Executor:
    def __new__(cls, env, policy, calibration=None, optimizer=None, evaluator=True):
        if env.task == "sorting_basic":
            from .runtime import TaskExecutor

            return TaskExecutor(env, policy, calibration, optimizer, evaluator)
        from .legacy_execution import M1Executor

        if optimizer is not None:
            raise ValueError("Local optimizers are only available in the new task runtime")
        return M1Executor(env, policy, calibration)
