from __future__ import annotations

from .generic import GenericApplier


class LeverApplier(GenericApplier):
    async def run(self) -> dict:
        await self.log("lever_detected", "Detected Lever application.")
        await self.log("screening_ready", "Preparing common Lever screening answers.")
        return await super().run()
