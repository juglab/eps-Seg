import torch.optim as optim
from torch.optim import Optimizer

class LabelSizeScheduler:
    def __init__(self, initial_size, final_size=None, step_interval=None):
        """
        A scheduler for label size during training.

        Args:
            initial_size (int): Initial label size.
            final_size (int): Final label size (if mode is "linear" or "step").
            step_interval (int): Interval of steps for size change (used in "step" mode).
            mode (str): Mode of scheduling. Options:
                        - "constant": Keeps the label size constant.
                        - "step": Changes the label size in steps.
        """
        self.initial_size = initial_size
        self.final_size = final_size
        self.step_interval = step_interval
        if initial_size == final_size:
            self.mode = "constant"
        else:
            self.mode = "step"
        self.current_size = initial_size

    def get_label_size(self, current_step):
        """
        Get the current label size based on the mode and step.

        Args:
            current_step (int): Current training step.

        Returns:
            int: Current label size.
        """
        if self.mode == "constant":
            return self.initial_size

        elif self.mode == "step":
            if self.final_size is None or self.step_interval is None:
                raise ValueError("final_size and step_interval must be provided for 'step' mode.")
            # Step-based change
            direction = 2 if self.final_size > self.initial_size else -2
            intervals = current_step // self.step_interval
            new_size = self.initial_size + intervals * direction
            return max(
                min(self.initial_size, self.final_size), min(max(self.initial_size, self.final_size), new_size)
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self):
        """Reset the scheduler to its initial state."""
        self.current_size = self.initial_size