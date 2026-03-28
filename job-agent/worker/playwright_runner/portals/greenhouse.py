from __future__ import annotations

from .generic import GenericApplier


class GreenhouseApplier(GenericApplier):
    async def run(self) -> dict:
        await self.log("greenhouse_detected", "Detected Greenhouse application.")
        await self.log("attachments_ready", "Preparing resume and cover letter upload flow.")
        return await super().run()
