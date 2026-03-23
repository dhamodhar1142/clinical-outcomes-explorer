from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.export_utils import (
    apply_export_policy,
    apply_role_based_redaction,
    build_policy_aware_bundle_profile,
    build_report_support_tables,
    build_role_export_bundle_manifest,
    build_role_export_bundle_text,
    json_bytes,
    normalize_report_mode,
)
from src.modules.privacy_security import (
    apply_dataframe_redaction,
    apply_export_watermark,
    build_export_governance_summary,
)
from src.services.report_service import generate_report_deliverable


EXCEL_ROW_CAP = 25000
JSON_ROW_CAP = 5000
PDF_LINES_PER_PAGE = 44


@dataclass(frozen=True)
class ExportArtifactBundle:
    report_bytes: bytes
    pdf_bytes: bytes
    excel_bytes: bytes
    json_bytes_payload: bytes
    zip_bytes: bytes
    manifest: list[dict[str, Any]]
    export_strategy: dict[str, Any]


def _slug(value: str) -> str:
    cleaned = ''.join(char.lower() if char.isalnum() else '_' for char in str(value).strip())
    while '__' in cleaned:
        cleaned = cleaned.replace('__', '_')
    return cleaned.strip('_') or 'artifact'


def _safe_df(value: Any) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _normalize_report_file_stem(report_label: str) -> str:
    return _slug(report_label.replace('Report', '').replace('Summary', '').strip() or report_label)


def _export_strategy(pipeline: dict[str, Any]) -> dict[str, Any]:
    overview = dict(pipeline.get('overview', {}))
    row_count = int(overview.get('rows', 0) or 0)
    sample_info = dict(pipeline.get('sample_info', {}))
    sampling_applied = bool(sample_info.get('sampling_applied'))
    workbook_row_cap = min(EXCEL_ROW_CAP, max(500, row_count if row_count > 0 else EXCEL_ROW_CAP))
    json_row_cap = min(JSON_ROW_CAP, max(250, row_count if row_count > 0 else JSON_ROW_CAP))
    return {
        'row_count': row_count,
        'sampling_applied': sampling_applied,
        'excel_row_cap': workbook_row_cap,
        'json_row_cap': json_row_cap,
        'large_dataset_mode': row_count > 100000 or sampling_applied,
        'strategy_note': (
            'Large-dataset export mode is active. Export artifacts include summary-first tables and capped supporting rows.'
            if row_count > 100000 or sampling_applied
            else 'Standard export mode is active. Full summary artifacts are included for the current pipeline view.'
        ),
    }


def _limited_table(table: pd.DataFrame, row_cap: int) -> pd.DataFrame:
    safe_table = _safe_df(table)
    if safe_table.empty:
        return safe_table
    return safe_table.head(row_cap).copy()


def _build_support_tables(report_mode: str, pipeline: dict[str, Any], row_cap: int) -> dict[str, pd.DataFrame]:
    overview = dict(pipeline.get('overview', {}))
    overview.setdefault('rows', 0)
    overview.setdefault('columns', 0)
    overview.setdefault('duplicate_rows', 0)
    overview.setdefault('missing_values', 0)
    quality = dict(pipeline.get('quality', {}))
    quality.setdefault('high_missing', pd.DataFrame())
    readiness = dict(pipeline.get('readiness', {}))
    readiness.setdefault('readiness_table', pd.DataFrame())
    healthcare = dict(pipeline.get('healthcare', {}))
    tables = build_report_support_tables(
        report_mode,
        overview,
        quality,
        readiness,
        healthcare,
        _safe_df(pipeline.get('action_recommendations')),
    )
    limited = {
        section: _limited_table(table, row_cap)
        for section, table in tables.items()
        if isinstance(table, pd.DataFrame)
    }
    overview_table = pd.DataFrame(
        [
            {'metric': 'Dataset', 'value': pipeline.get('dataset_name', 'Current dataset')},
            {'metric': 'Rows', 'value': int(pipeline.get('overview', {}).get('rows', 0) or 0)},
            {'metric': 'Analyzed columns', 'value': int(pipeline.get('overview', {}).get('analyzed_columns', pipeline.get('overview', {}).get('columns', 0)) or 0)},
            {'metric': 'Quality score', 'value': float(pipeline.get('quality', {}).get('quality_score', 0.0) or 0.0)},
            {'metric': 'Readiness score', 'value': float(pipeline.get('readiness', {}).get('readiness_score', 0.0) or 0.0)},
            {'metric': 'Healthcare readiness score', 'value': float(pipeline.get('healthcare', {}).get('healthcare_readiness_score', 0.0) or 0.0)},
        ]
    )
    limited.setdefault('Export Overview', overview_table)
    return limited


def _apply_export_protection(
    payload: bytes,
    *,
    role: str,
    policy_name: str | None,
    privacy_review: dict[str, Any] | None,
) -> bytes:
    protected = payload
    if policy_name:
        protected = apply_export_policy(protected, policy_name, privacy_review or {})
    return apply_role_based_redaction(protected, role, privacy_review or {})


def _apply_export_table_protection(
    table: pd.DataFrame,
    *,
    role: str,
    privacy_review: dict[str, Any] | None,
    redaction_level: str,
) -> pd.DataFrame:
    return apply_dataframe_redaction(
        table,
        privacy_review=privacy_review or {},
        redaction_level=redaction_level,
        role=role,
    )


def _pdf_escape(value: str) -> str:
    return value.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def _xml_escape(value: Any) -> str:
    text = str(value if value is not None else '')
    return (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;')
    )


def _build_pdf_from_text(title: str, text: str) -> bytes:
    source_lines = [title, ''] + text.splitlines()
    pages = [source_lines[idx: idx + PDF_LINES_PER_PAGE] for idx in range(0, len(source_lines), PDF_LINES_PER_PAGE)] or [[]]
    objects: list[bytes] = []

    def _append_object(payload: str | bytes) -> int:
        data = payload.encode('latin-1', errors='replace') if isinstance(payload, str) else payload
        objects.append(data)
        return len(objects)

    catalog_id = _append_object('')
    pages_id = _append_object('')
    font_id = _append_object('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>')
    page_ids: list[int] = []

    for lines in pages:
        content_lines = ['BT', '/F1 10 Tf', '50 780 Td', '14 TL']
        for line in lines:
            content_lines.append(f'({_pdf_escape(line)}) Tj')
            content_lines.append('T*')
        content_lines.append('ET')
        content_body = '\n'.join(content_lines).encode('latin-1', errors='replace')
        content_id = _append_object(
            b'<< /Length ' + str(len(content_body)).encode('ascii') + b' >>\nstream\n' + content_body + b'\nendstream'
        )
        page_id = _append_object(
            f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] '
            f'/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>'
        )
        page_ids.append(page_id)

    objects[catalog_id - 1] = f'<< /Type /Catalog /Pages {pages_id} 0 R >>'.encode('ascii')
    kids = ' '.join(f'{page_id} 0 R' for page_id in page_ids)
    objects[pages_id - 1] = f'<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>'.encode('ascii')

    buffer = BytesIO()
    buffer.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f'{index} 0 obj\n'.encode('ascii'))
        buffer.write(obj)
        buffer.write(b'\nendobj\n')
    xref_offset = buffer.tell()
    buffer.write(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
    buffer.write(b'0000000000 65535 f \n')
    for offset in offsets[1:]:
        buffer.write(f'{offset:010d} 00000 n \n'.encode('ascii'))
    buffer.write(
        f'trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_offset}\n%%EOF'.encode('ascii')
    )
    return buffer.getvalue()


def _xlsx_column_name(index: int) -> str:
    label = ''
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        label = chr(65 + remainder) + label
    return label or 'A'


def _sheet_xml_rows(rows: list[list[Any]]) -> tuple[str, dict[int, int]]:
    xml_rows: list[str] = []
    widths: dict[int, int] = {}
    for row_index, row in enumerate(rows, start=1):
        xml_cells: list[str] = []
        for column_index, value in enumerate(row, start=1):
            cell_ref = f'{_xlsx_column_name(column_index)}{row_index}'
            style_id = '1' if row_index == 1 else '0'
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cell_xml = f'<c r="{cell_ref}" s="{style_id}"><v>{value}</v></c>'
            else:
                text = _xml_escape(value)
                cell_xml = (
                    f'<c r="{cell_ref}" s="{style_id}" t="inlineStr">'
                    f'<is><t xml:space="preserve">{text}</t></is></c>'
                )
            widths[column_index] = min(max(widths.get(column_index, 8), len(str(value if value is not None else '')) + 2), 60)
            xml_cells.append(cell_xml)
        xml_rows.append(f'<row r="{row_index}">{"".join(xml_cells)}</row>')
    return ''.join(xml_rows), widths


def _sheet_xml(title: str, rows: list[list[Any]]) -> str:
    row_xml, widths = _sheet_xml_rows(rows)
    cols_xml = ''.join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in widths.items()
    )
    pane_xml = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'{pane_xml}'
        f'<cols>{cols_xml}</cols>'
        f'<sheetData>{row_xml}</sheetData>'
        '</worksheet>'
    )


def _build_excel_workbook(
    *,
    dataset_name: str,
    report_label: str,
    report_text: str,
    support_tables: dict[str, pd.DataFrame],
    manifest_df: pd.DataFrame,
    export_strategy: dict[str, Any],
    governance_summary: dict[str, Any],
) -> bytes:
    overview_rows = [
        ('Dataset', dataset_name),
        ('Report label', report_label),
        ('Rows', export_strategy.get('row_count', 0)),
        ('Large dataset mode', 'Yes' if export_strategy.get('large_dataset_mode') else 'No'),
        ('Sampling applied', 'Yes' if export_strategy.get('sampling_applied') else 'No'),
        ('Excel row cap', export_strategy.get('excel_row_cap', EXCEL_ROW_CAP)),
        ('JSON row cap', export_strategy.get('json_row_cap', JSON_ROW_CAP)),
        ('Strategy note', export_strategy.get('strategy_note', '')),
    ]
    sheets: list[tuple[str, list[list[Any]]]] = [('Overview', [['Field', 'Value'], *overview_rows])]
    governance_rows = [
        ['Control', 'Value'],
        ['Policy', governance_summary.get('policy_evaluation', {}).get('policy_name', 'Internal Review')],
        ['Redaction Level', governance_summary.get('policy_evaluation', {}).get('redaction_level', 'Low')],
        ['Workspace Export Access', governance_summary.get('policy_evaluation', {}).get('workspace_export_access', 'Editors and owners')],
        ['Workspace Export Role', governance_summary.get('policy_evaluation', {}).get('workspace_export_role', 'Viewer')],
        ['Watermark', governance_summary.get('policy_evaluation', {}).get('watermark_label', '')],
    ]
    sheets.append(('Governance', governance_rows))
    sheets.append(('Report Preview', [['Generated report text'], *[[line] for line in (report_text.splitlines() or [''])]]))
    manifest_rows = [list(manifest_df.columns)] + manifest_df.fillna('').values.tolist() if not manifest_df.empty else [['status'], ['No rows available']]
    sheets.append(('Bundle Manifest', manifest_rows))
    for section, table in support_tables.items():
        safe_table = _safe_df(table)
        if safe_table.empty:
            rows = [['status'], ['No rows available']]
        else:
            rows = [list(safe_table.columns)] + safe_table.fillna('').values.tolist()
        sheets.append((section[:31] or 'Sheet', rows))

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        + ''.join(
            f'<sheet name="{_xml_escape(title)}" sheetId="{index}" r:id="rId{index}"/>'
            for index, (title, _) in enumerate(sheets, start=1)
        )
        + '</sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + ''.join(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index, _sheet in enumerate(sheets, start=1)
        )
        + f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        + '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="1" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + ''.join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index, _sheet in enumerate(sheets, start=1)
        )
        + '</Types>'
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    buffer = BytesIO()
    with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr('[Content_Types].xml', content_types_xml)
        workbook_zip.writestr('_rels/.rels', root_rels_xml)
        workbook_zip.writestr('xl/workbook.xml', workbook_xml)
        workbook_zip.writestr('xl/_rels/workbook.xml.rels', workbook_rels_xml)
        workbook_zip.writestr('xl/styles.xml', styles_xml)
        for index, (title, rows) in enumerate(sheets, start=1):
            workbook_zip.writestr(f'xl/worksheets/sheet{index}.xml', _sheet_xml(title, rows))
    return buffer.getvalue()


def _build_json_payload(
    *,
    dataset_name: str,
    source_mode: str,
    report_label: str,
    report_text: str,
    support_tables: dict[str, pd.DataFrame],
    manifest_df: pd.DataFrame,
    export_strategy: dict[str, Any],
    governance_summary: dict[str, Any],
) -> bytes:
    payload = {
        'dataset_name': dataset_name,
        'source_mode': source_mode,
        'report_label': report_label,
        'report_text': report_text,
        'export_strategy': export_strategy,
        'governance': {
            'policy_name': governance_summary.get('policy_evaluation', {}).get('policy_name', 'Internal Review'),
            'redaction_level': governance_summary.get('policy_evaluation', {}).get('redaction_level', 'Low'),
            'workspace_export_access': governance_summary.get('policy_evaluation', {}).get('workspace_export_access', 'Editors and owners'),
            'workspace_export_role': governance_summary.get('policy_evaluation', {}).get('workspace_export_role', 'Viewer'),
            'watermark_label': governance_summary.get('policy_evaluation', {}).get('watermark_label', ''),
            'classification_summary': governance_summary.get('policy_evaluation', {}).get('classification_summary', {}),
        },
        'bundle_manifest': manifest_df.to_dict(orient='records'),
        'support_tables': {
            section: _limited_table(table, int(export_strategy.get('json_row_cap', JSON_ROW_CAP))).to_dict(orient='records')
            for section, table in support_tables.items()
        },
    }
    return json_bytes(payload)


def _build_zip_bundle(
    *,
    dataset_name: str,
    report_label: str,
    report_bytes: bytes,
    pdf_bytes: bytes,
    excel_bytes: bytes,
    json_payload_bytes: bytes,
    bundle_text: bytes,
) -> tuple[bytes, list[dict[str, Any]]]:
    stem = _normalize_report_file_stem(report_label)
    base_name = _slug(dataset_name)
    files = [
        (f'{base_name}_{stem}.txt', report_bytes, 'text/plain'),
        (f'{base_name}_{stem}.pdf', pdf_bytes, 'application/pdf'),
        (f'{base_name}_{stem}.xlsx', excel_bytes, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        (f'{base_name}_{stem}.json', json_payload_bytes, 'application/json'),
        (f'{base_name}_{stem}_bundle_guide.txt', bundle_text, 'text/plain'),
    ]
    buffer = BytesIO()
    with ZipFile(buffer, mode='w', compression=ZIP_DEFLATED) as archive:
        for file_name, payload, _ in files:
            archive.writestr(file_name, payload)
    manifest = [
        {'file_name': file_name, 'mime_type': mime_type, 'size_bytes': len(payload)}
        for file_name, payload, mime_type in files
    ]
    manifest.append(
        {
            'file_name': f'{base_name}_{stem}_bundle.zip',
            'mime_type': 'application/zip',
            'size_bytes': len(buffer.getvalue()),
        }
    )
    return buffer.getvalue(), manifest


def prepare_policy_aware_export_bundle(
    *,
    role: str,
    report_mode: str,
    policy_name: str,
    export_allowed: bool,
    privacy_review: dict[str, Any],
    workspace_identity: dict[str, Any] | None = None,
    governance_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    governance_summary = build_export_governance_summary(
        policy_name,
        privacy_review,
        workspace_identity,
        governance_config,
    )
    bundle_title, bundle_table = build_policy_aware_bundle_profile(
        role,
        report_mode,
        policy_name,
        export_allowed and bool(governance_summary.get('policy_evaluation', {}).get('workspace_export_allowed', True)),
        privacy_review,
    )
    bundle_manifest = _apply_export_table_protection(
        build_role_export_bundle_manifest(
            role,
            policy_name,
            export_allowed and bool(governance_summary.get('policy_evaluation', {}).get('workspace_export_allowed', True)),
            report_mode,
            privacy_review,
        ),
        role=role,
        privacy_review=privacy_review,
        redaction_level=str(governance_summary.get('policy_evaluation', {}).get('redaction_level', 'Low')),
    )
    bundle_text = _apply_export_protection(
        build_role_export_bundle_text(
            role,
            policy_name,
            export_allowed and bool(governance_summary.get('policy_evaluation', {}).get('workspace_export_allowed', True)),
            report_mode,
            privacy_review,
        ),
        role=role,
        policy_name=policy_name,
        privacy_review=privacy_review,
    )
    bundle_text = apply_export_watermark(
        bundle_text,
        str(governance_summary.get('policy_evaluation', {}).get('watermark_label', 'Internal export')),
    )
    return {
        'bundle_title': bundle_title,
        'bundle_table': bundle_table,
        'bundle_manifest': bundle_manifest,
        'bundle_text': bundle_text,
        'governance_summary': governance_summary,
    }


def record_export_bundle_metadata(
    application_service: Any | None,
    workspace_identity: dict[str, Any] | None,
    *,
    dataset_name: str,
    bundle_label: str,
    file_name: str,
    artifact_path: str = '',
) -> None:
    if application_service is None:
        return
    application_service.record_report_metadata(
        workspace_identity or {},
        {
            'dataset_name': dataset_name,
            'report_type': bundle_label,
            'file_name': file_name,
            'artifact_path': artifact_path,
            'status': 'generated',
        },
    )


def record_export_bundle_metadata_once(
    session_state: dict[str, Any],
    application_service: Any | None,
    workspace_identity: dict[str, Any] | None,
    *,
    dataset_name: str,
    bundle_label: str,
    file_name: str,
    artifact_path: str = '',
) -> None:
    recorded = session_state.setdefault('recorded_export_bundle_metadata', set())
    key = (str((workspace_identity or {}).get('workspace_id', 'guest-demo-workspace')), dataset_name, bundle_label, file_name)
    if key in recorded:
        return
    record_export_bundle_metadata(
        application_service,
        workspace_identity,
        dataset_name=dataset_name,
        bundle_label=bundle_label,
        file_name=file_name,
        artifact_path=artifact_path,
    )
    recorded.add(key)


def generate_export_report_output(
    session_state: dict[str, Any],
    *,
    job_runtime: dict[str, Any],
    report_label: str,
    dataset_name: str,
    pipeline: dict[str, Any],
    workspace_identity: dict[str, Any] | None = None,
    role: str = 'Analyst',
    progress_callback=None,
    application_service: Any | None = None,
    storage_service: Any | None = None,
    policy_name: str = 'Internal Review',
    privacy_review: dict[str, Any] | None = None,
    governance_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_mode_lookup = {
        'Executive Report': 'Executive Summary',
        'Analyst Report': 'Analyst Report',
        'Data Readiness Report': 'Data Readiness Review',
        'Clinical Summary': 'Clinical Report',
        'Readmission Report': 'Clinical Report',
        'Dataset Summary Report': 'Analyst Report',
    }
    report_mode = normalize_report_mode(normalized_mode_lookup.get(report_label, report_label))
    effective_privacy = privacy_review or pipeline.get('privacy_review', {})
    bundle_profile = prepare_policy_aware_export_bundle(
        role=role,
        report_mode=report_mode,
        policy_name=policy_name,
        export_allowed=True,
        privacy_review=effective_privacy,
        workspace_identity=workspace_identity,
        governance_config=governance_config,
    )
    governance_summary = dict(bundle_profile.get('governance_summary', {}))
    if not bool(governance_summary.get('policy_evaluation', {}).get('workspace_export_allowed', True)):
        raise PermissionError(
            'The active workspace export access policy restricts this export for the current workspace role.'
        )

    if progress_callback:
        progress_callback(0.05, 'Preparing report generation job...')
    report_result = generate_report_deliverable(
        session_state,
        job_runtime=job_runtime,
        report_label=report_label,
        dataset_name=dataset_name,
        pipeline=pipeline,
        workspace_identity=workspace_identity,
        role=role,
        progress_callback=progress_callback,
    )
    report_bytes = report_result.get('result', b'') or b''
    protected_report_bytes = _apply_export_protection(
        report_bytes,
        role=role,
        policy_name=policy_name,
        privacy_review=effective_privacy,
    )
    protected_report_bytes = apply_export_watermark(
        protected_report_bytes,
        str(governance_summary.get('policy_evaluation', {}).get('watermark_label', 'Internal export')),
    )
    report_text = protected_report_bytes.decode('utf-8', errors='replace')
    export_strategy = _export_strategy(pipeline)
    support_tables = {
        section: _apply_export_table_protection(
            table,
            role=role,
            privacy_review=effective_privacy,
            redaction_level=str(governance_summary.get('policy_evaluation', {}).get('redaction_level', 'Low')),
        )
        for section, table in _build_support_tables(
            report_mode,
            {**pipeline, 'dataset_name': dataset_name},
            int(export_strategy['excel_row_cap']),
        ).items()
    }

    if progress_callback:
        progress_callback(0.70, 'Rendering PDF and workbook artifacts...')
    pdf_bytes = _build_pdf_from_text(report_label, report_text)
    excel_bytes = _build_excel_workbook(
        dataset_name=dataset_name,
        report_label=report_label,
        report_text=report_text,
        support_tables=support_tables,
        manifest_df=bundle_profile['bundle_manifest'],
        export_strategy=export_strategy,
        governance_summary=governance_summary,
    )

    if progress_callback:
        progress_callback(0.85, 'Building JSON payload and ZIP bundle...')
    json_payload_bytes = _build_json_payload(
        dataset_name=dataset_name,
        source_mode=str(pipeline.get('source_meta', {}).get('source_mode', 'Unknown')),
        report_label=report_label,
        report_text=report_text,
        support_tables=support_tables,
        manifest_df=bundle_profile['bundle_manifest'],
        export_strategy=export_strategy,
        governance_summary=governance_summary,
    )
    zip_bytes, artifact_manifest = _build_zip_bundle(
        dataset_name=dataset_name,
        report_label=report_label,
        report_bytes=protected_report_bytes,
        pdf_bytes=pdf_bytes,
        excel_bytes=excel_bytes,
        json_payload_bytes=json_payload_bytes,
        bundle_text=bundle_profile['bundle_text'],
    )

    artifact = None
    if storage_service is not None and bool(getattr(storage_service, 'enabled', False)):
        artifact = storage_service.save_report_artifact(
            workspace_identity,
            dataset_name=dataset_name,
            report_type=report_label,
            file_name=f'{_normalize_report_file_stem(report_label)}_bundle.zip',
            payload=zip_bytes,
        )

    metadata_path = str((artifact or {}).get('artifact_path', ''))
    record_export_bundle_metadata(
        application_service,
        workspace_identity,
        dataset_name=dataset_name,
        bundle_label=report_label,
        file_name=f'{_normalize_report_file_stem(report_label)}_bundle.zip',
        artifact_path=metadata_path,
    )

    if progress_callback:
        progress_callback(1.0, f'{report_label} export bundle is ready.')

    export_bundle = ExportArtifactBundle(
        report_bytes=protected_report_bytes,
        pdf_bytes=pdf_bytes,
        excel_bytes=excel_bytes,
        json_bytes_payload=json_payload_bytes,
        zip_bytes=zip_bytes,
        manifest=artifact_manifest,
        export_strategy=export_strategy,
    )
    return {
        'job': report_result.get('job'),
        'status': report_result.get('status', {}),
        'result': protected_report_bytes,
        'report_bytes': protected_report_bytes,
        'pdf_bytes': pdf_bytes,
        'excel_bytes': excel_bytes,
        'json_bytes': json_payload_bytes,
        'zip_bytes': zip_bytes,
        'bundle_manifest': pd.DataFrame(artifact_manifest),
        'export_bundle': export_bundle,
        'artifact': artifact,
        'support_tables': support_tables,
        'bundle_profile': bundle_profile,
        'governance_summary': governance_summary,
        'export_strategy': export_strategy,
    }


__all__ = [
    'ExportArtifactBundle',
    'generate_export_report_output',
    'prepare_policy_aware_export_bundle',
    'record_export_bundle_metadata',
    'record_export_bundle_metadata_once',
]
