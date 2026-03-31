from __future__ import annotations

from ..utils.base import BaseApplier


class GenericApplier(BaseApplier):
    async def run(self) -> dict:
        await self.log("open_apply", "Navigate to the apply URL.")
        for step in self.plan.get("steps", []):
            await self.log(step.get("step_id", "step"), step.get("description", ""))
        await self.log("review_ready", "Stopped before submit because review-first mode is active.")
        return {
            "status": "review_ready",
            "portal_type": self.plan.get("portal_type", "generic"),
            "logs": [log.__dict__ for log in self.logs],
        }
