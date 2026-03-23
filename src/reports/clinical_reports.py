from __future__ import annotations

from io import StringIO

import pandas as pd

from src.reports.common import _analyzed_columns, _source_columns


def build_text_report(dataset_name: str, overview: dict[str, object], structure, field_profile: pd.DataFrame, quality: dict[str, object], semantic: dict[str, object], readiness: dict[str, object], healthcare: dict[str, object], insights: dict[str, object]) -> bytes:
    buffer = StringIO()
    buffer.write('Smart Dataset Analyzer Summary\n')
    buffer.write('=' * 32 + '\n\n')
    buffer.write(f'Dataset: {dataset_name}\n')
    buffer.write(f"Rows: {overview['rows']:,}\n")
    buffer.write(f"Analyzed columns: {_analyzed_columns(overview):,}\n")
    if _source_columns(overview) != _analyzed_columns(overview):
        buffer.write(f"Source columns: {_source_columns(overview):,}\n")
    buffer.write(f"Duplicate rows: {overview['duplicate_rows']:,}\n")
    buffer.write(f"Missing values: {overview['missing_values']:,}\n")
    buffer.write(f"Memory estimate: {overview['memory_mb']:.2f} MB\n\n")

    buffer.write('Detected Column Types\n')
    buffer.write('-' * 22 + '\n')
    for _, row in structure.detection_table.iterrows():
        buffer.write(f"{row['column_name']}: {row['inferred_type']} ({row['confidence_score']:.2f})\n")
    buffer.write('\n')

    buffer.write('Semantic Mappings\n')
    buffer.write('-' * 18 + '\n')
    mapping_table = semantic['mapping_table']
    if mapping_table.empty:
        buffer.write('No strong semantic mappings were detected.\n')
    else:
        for _, row in mapping_table.iterrows():
            buffer.write(f"{row['original_column']} -> {row['semantic_label']} [{row['confidence_label']}]\n")
    buffer.write('\n')

    buffer.write('Data Quality Notes\n')
    buffer.write('-' * 18 + '\n')
    if not quality['high_missing'].empty:
        buffer.write('High missingness columns:\n')
        for _, row in quality['high_missing'].head(5).iterrows():
            buffer.write(f"- {row['column_name']}: {row['null_percentage']:.1%}\n")
    else:
        buffer.write('No major missingness issues were detected.\n')
    buffer.write('\n')

    buffer.write('Analysis Readiness\n')
    buffer.write('-' * 18 + '\n')
    for _, row in readiness['readiness_table'].iterrows():
        buffer.write(f"- {row['analysis_module']}: {row['status']}\n")
    buffer.write('\n')

    buffer.write('Key Insights\n')
    buffer.write('-' * 12 + '\n')
    for line in insights['summary_lines']:
        buffer.write(f"- {line}\n")
    buffer.write('\nRecommendations\n')
    buffer.write('-' * 15 + '\n')
    for line in insights['recommendations']:
        buffer.write(f"- {line}\n")
    if 'bmi_original_value' in field_profile['column_name'].astype(str).tolist() or 'bmi' in field_profile['column_name'].astype(str).tolist():
        buffer.write('\nRemediation Notes\n')
        buffer.write('-' * 17 + '\n')
        if 'cost_amount' in semantic.get('canonical_map', {}):
            buffer.write('- Financial analysis may include a synthetic cost field when no native source cost was available.\n')
        if 'diagnosis_code' in semantic.get('canonical_map', {}):
            buffer.write('- Clinical segmentation may include derived diagnosis labels for demo-safe analytics support.\n')
        if 'readmission' in semantic.get('canonical_map', {}):
            buffer.write('- Readmission analytics may rely on a synthetic or inferred support field when native readmission flags are unavailable.\n')

    return buffer.getvalue().encode('utf-8')

