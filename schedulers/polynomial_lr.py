"""
Polynomial learning-rate scheduler.

Implements the standard polynomial-decay schedule:

    lr(t) = max( initial_lr x (1 - t / T)^p , min_lr )

where
    t = current step,  T = total_steps,  p = power.

Special cases
  * power = 1.0  ->  linear decay
  * power = 2.0  ->  quadratic decay (slower initial drop)
  * power = 0.5  ->  square-root decay (faster initial drop)

The scheduler is typically used by calling :meth:`step` once per training
iteration and passing the returned learning rate to the solver.
"""


class PolynomialLR:
    """Polynomial learning-rate decay scheduler.

    Parameters
    ----------
    initial_lr : float
        Starting learning rate.
    total_steps : int
        Total number of training steps over which the learning rate decays.
    power : float, default 1.0
        Exponent of the polynomial.  ``power = 1`` gives linear decay.
    min_lr : float, default 0.0
        Floor value - the learning rate will never drop below this.

    Examples
    --------
    >>> sched = PolynomialLR(initial_lr=0.01, total_steps=100, power=2.0)
    >>> sched.get_lr()
    0.01
    >>> sched.step()     # advance one step, returns new lr
    0.009801
    """

    def __init__(self, initial_lr, total_steps, power=1.0, min_lr=0.0):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if initial_lr < 0:
            raise ValueError("initial_lr must be non-negative")

        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        self.current_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr(self):
        """Return the learning rate for the current step.

        Returns
        -------
        lr : float
            Current learning rate, clamped to [min_lr, initial_lr].
        """
        if self.current_step >= self.total_steps:
            return self.min_lr

        # Polynomial decay factor
        decay = (1.0 - self.current_step / self.total_steps) ** self.power
        lr = self.initial_lr * decay

        # Clamp to minimum
        return max(lr, self.min_lr)

    def step(self):
        """Advance the internal counter by one and return the new learning rate.

        Returns
        -------
        lr : float
            Learning rate **after** incrementing the step counter.
        """
        self.current_step += 1
        return self.get_lr()

    def reset(self):
        """Reset the step counter to zero (e.g. for warm restarts)."""
        self.current_step = 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"PolynomialLR(initial_lr={self.initial_lr}, "
            f"total_steps={self.total_steps}, "
            f"power={self.power}, min_lr={self.min_lr}, "
            f"current_step={self.current_step})"
        )
