from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

NOTE_LABEL_SEPARATOR = " | "


def _build_note_target_label(target_type: str, target_name: str) -> str:
    return f"{target_type}{NOTE_LABEL_SEPARATOR}{target_name}"


def _parse_note_target_label(target_label: str) -> tuple[str, str] | None:
    if NOTE_LABEL_SEPARATOR in target_label:
        parts = target_label.split(NOTE_LABEL_SEPARATOR, 1)
        return parts[0], parts[1]
    legacy_parts = target_label.split(" · ", 1)
    if len(legacy_parts) == 2:
        return legacy_parts[0], legacy_parts[1]
    return None

def build_note_target_options(
    dataset_name: str,
    workflow_pack_names: list[str],
    report_modes: list[str],
    section_labels: list[str],
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = [
        {
            "target_type": "Dataset",
            "target_name": dataset_name or "Current dataset",
            "label": _build_note_target_label("Dataset", dataset_name or "Current dataset"),
        }
    ]
    for pack_name in workflow_pack_names:
        options.append(
            {
                "target_type": "Workflow Pack",
                "target_name": pack_name,
                "label": _build_note_target_label("Workflow Pack", pack_name),
            }
        )
    for report_mode in report_modes:
        options.append(
            {
                "target_type": "Report",
                "target_name": report_mode,
                "label": _build_note_target_label("Report", report_mode),
            }
        )
    for section in section_labels:
        options.append(
            {
                "target_type": "Analysis Section",
                "target_name": section,
                "label": _build_note_target_label("Analysis Section", section),
            }
        )
    return options


def append_collaboration_note(
    notes_store: list[dict[str, Any]],
    target_type: str,
    target_name: str,
    note_text: str,
    author: str,
    workspace_name: str,
    section_name: str = "",
    author_user_id: str = "",
    workspace_id: str = "",
    workspace_role: str = "",
) -> list[dict[str, Any]]:
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_type": target_type,
        "target_name": target_name,
        "section_name": section_name or target_name,
        "author": author or "Local Demo User",
        "author_user_id": author_user_id or "guest-user",
        "workspace_name": workspace_name or "Guest Demo Workspace",
        "workspace_id": workspace_id or "guest-demo-workspace",
        "workspace_role": workspace_role or "viewer",
        "note_text": note_text.strip(),
    }
    notes_store.append(entry)
    return notes_store


def build_collaboration_notes_view(
    notes_store: list[dict[str, Any]],
    target_label: str = "All Targets",
) -> pd.DataFrame:
    notes = pd.DataFrame(notes_store)
    if notes.empty:
        return pd.DataFrame(columns=["timestamp", "target_type", "target_name", "section_name", "author", "note_text"])
    if target_label != "All Targets":
        parts = _parse_note_target_label(target_label)
        if parts:
            target_type, target_name = parts
            notes = notes[
                (notes["target_type"].astype(str) == target_type)
                & (notes["target_name"].astype(str) == target_name)
            ]
    return notes.sort_values("timestamp", ascending=False).reset_index(drop=True)


def build_collaboration_note_summary(notes_store: list[dict[str, Any]]) -> dict[str, Any]:
    notes = pd.DataFrame(notes_store)
    if notes.empty:
        return {
            "summary_cards": [
                {"label": "Notes", "value": "0"},
                {"label": "Datasets", "value": "0"},
                {"label": "Reports", "value": "0"},
                {"label": "Workflow Packs", "value": "0"},
            ],
            "history": pd.DataFrame(columns=["target_type", "notes"]),
            "recent_notes": pd.DataFrame(columns=["timestamp", "target_type", "target_name", "author", "note_text"]),
            "top_targets": pd.DataFrame(columns=["target_label", "notes", "last_updated"]),
        }
    history = (
        notes.groupby("target_type")
        .size()
        .reset_index(name="notes")
        .sort_values("notes", ascending=False)
        .reset_index(drop=True)
    )
    recent_notes = notes.sort_values("timestamp", ascending=False).reset_index(drop=True).head(8)
    top_targets = (
        notes.assign(target_label=notes["target_type"].astype(str) + NOTE_LABEL_SEPARATOR + notes["target_name"].astype(str))
        .groupby("target_label")
        .agg(notes=("note_text", "size"), last_updated=("timestamp", "max"))
        .reset_index()
        .sort_values(["notes", "last_updated"], ascending=[False, False])
        .reset_index(drop=True)
        .head(8)
    )
    return {
        "summary_cards": [
            {"label": "Notes", "value": f"{len(notes):,}"},
            {"label": "Datasets", "value": f"{int((notes['target_type'] == 'Dataset').sum()):,}"},
            {"label": "Reports", "value": f"{int((notes['target_type'] == 'Report').sum()):,}"},
            {"label": "Workflow Packs", "value": f"{int((notes['target_type'] == 'Workflow Pack').sum()):,}"},
        ],
        "history": history,
        "recent_notes": recent_notes[["timestamp", "target_type", "target_name", "author", "note_text"]],
        "top_targets": top_targets,
    }
