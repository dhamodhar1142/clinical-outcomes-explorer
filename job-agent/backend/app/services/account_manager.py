from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AccountManager:
    def __init__(self) -> None:
        self.path = Path(os.getenv("JOB_AGENT_ACCOUNTS_PATH", "job-agent/data/accounts.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load_accounts(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def find_account(self, company: str, portal_type: str) -> dict[str, Any] | None:
        for account in self.load_accounts():
            if account.get("company", "").lower() == company.lower() and account.get("portal_type") == portal_type:
                return account
        return None

    def save_account(self, company: str, portal_type: str, username: str, secret_ref: str) -> dict[str, Any]:
        accounts = self.load_accounts()
        record = {
            "company": company,
            "portal_type": portal_type,
            "username": username,
            "secret_ref": secret_ref,
        }
        accounts = [item for item in accounts if not (item.get("company") == company and item.get("portal_type") == portal_type)]
        accounts.append(record)
        self.path.write_text(json.dumps(accounts, indent=2), encoding="utf-8")
        return record
