from tensordict import TensorDictBase
from torchrl.envs import StepCounter


class CustomStepCounter(StepCounter):
    def __init__(self, set_terminated=False, **kwargs):
        super().__init__(**kwargs)
        self.set_terminated = set_terminated

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        for step_count_key, truncated_key, done_key, terminated_key in zip(
            self.step_count_keys,
            self.truncated_keys,
            self.done_keys,
            self.terminated_keys,
        ):
            step_count = tensordict.get(step_count_key)
            next_step_count = step_count + 1
            next_tensordict.set(step_count_key, next_step_count)

            if self.max_steps is not None:
                truncated = next_step_count >= self.max_steps
                truncated = truncated | next_tensordict.get(truncated_key, False)
                if self.update_done:
                    done = next_tensordict.get(done_key, None)
                    terminated = next_tensordict.get(terminated_key, None)
                    if terminated is not None and not self.set_terminated:
                        truncated = truncated & ~terminated
                    done = truncated | done  # we assume no done after reset
                    next_tensordict.set(done_key, done)
                next_tensordict.set(truncated_key, truncated)
        return next_tensordict
