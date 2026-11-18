"""Decide when a background job may use the CPU: start only when the machine is
idle, stop as soon as someone else needs it.

The caller samples the CPU use of everything *except* the job (system total
minus the job's own share) every few seconds and feeds it to IdleGate.observe.
Keeping the decision in this small pure class lets it be tested without a busy
machine; tools/backfill_regions.py does the sampling and the process control.
"""

from __future__ import annotations

from dataclasses import dataclass, field

START = "start"
STOP = "stop"


@dataclass
class IdleGate:
    """Hysteresis on other programs' CPU use, in percent of the whole machine.

    start_below / settle_samples: the job may start after this many consecutive
    samples below start_below. stop_above / stop_samples: a running job must stop
    after this many consecutive samples above stop_above. A single spike (a window
    opening) does not stop the job; a sustained load does.
    """

    start_below: float = 25.0
    stop_above: float = 40.0
    settle_samples: int = 6
    stop_samples: int = 2
    _quiet: int = field(default=0, init=False)
    _busy: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.start_below > self.stop_above:
            raise ValueError("start_below must not exceed stop_above")
        if self.settle_samples < 1 or self.stop_samples < 1:
            raise ValueError("sample counts must be at least 1")

    def observe(self, others_percent: float, running: bool):
        """Feed one sample; return START, STOP or None (keep the current state)."""
        if running:
            self._quiet = 0
            self._busy = self._busy + 1 if others_percent > self.stop_above else 0
            if self._busy >= self.stop_samples:
                self._busy = 0
                return STOP
            return None
        self._busy = 0
        self._quiet = self._quiet + 1 if others_percent < self.start_below else 0
        if self._quiet >= self.settle_samples:
            self._quiet = 0
            return START
        return None


def others_percent(
    system_percent: float, job_percent_of_one_core: float, cpu_count: int
) -> float:
    """CPU use of everything except the job, in percent of the whole machine.

    psutil reports a process's use relative to one core (up to cpu_count * 100)
    and the system's relative to all cores (up to 100).
    """
    job = job_percent_of_one_core / max(1, cpu_count)
    return max(0.0, system_percent - job)
