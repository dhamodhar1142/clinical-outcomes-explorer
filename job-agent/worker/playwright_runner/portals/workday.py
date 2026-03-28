from __future__ import annotations

from .generic import GenericApplier


class WorkdayApplier(GenericApplier):
    async def run(self) -> dict:
        await self.log("workday_detected", "Detected Workday flow.")
        await self.log("workday_auth_gate", "Checking login or account creation requirements.")
        return await super().run()
