from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DATASET_DIR = ROOT / 'tests' / 'fixtures' / 'datasets'


@dataclass(frozen=True)
class DatasetFixture:
    key: str
    path: Path
    dataset_name: str
    dataset_type: str
    expected_key_source_columns: tuple[str, ...]
    expected_date_columns: tuple[str, ...]
    expected_identifier_fields: tuple[str, ...]
    expected_categorical_fields: tuple[str, ...]
    expected_approximate_row_count: int
    large_file: bool = False
    notes: str = ''


PRIMARY_FULL_VALIDATION_FIXTURE = DatasetFixture(
    key='primary-healthcare',
    path=FIXTURE_DATASET_DIR / 'STG_EHP__VIST.csv',
    dataset_name='STG_EHP__VIST.csv',
    dataset_type='healthcare',
    expected_key_source_columns=(
        'REFR_NO',
        'PAT_ID',
        'MEDT_ID',
        'VIS_EN',
        'VIS_EX',
        'VSTAT_CD',
        'VSTAT_DES',
        'VTYPE_CD',
        'VTYPE_DES',
        'ROM_ID',
    ),
    expected_date_columns=('VIS_EN', 'VIS_EX'),
    expected_identifier_fields=('REFR_NO', 'PAT_ID', 'MEDT_ID', 'ROM_ID'),
    expected_categorical_fields=('VSTAT_DES', 'VTYPE_DES', 'VSTAT_CD', 'VTYPE_CD'),
    expected_approximate_row_count=917331,
    large_file=True,
    notes='Primary full-workflow healthcare validation fixture. Large-file streaming path expected.',
)

SMALL_HEALTHCARE_FIXTURE = DatasetFixture(
    key='small-healthcare',
    path=FIXTURE_DATASET_DIR / 'SMALL_HEALTHCARE_VISITS.csv',
    dataset_name='SMALL_HEALTHCARE_VISITS.csv',
    dataset_type='healthcare',
    expected_key_source_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_key_source_columns,
    expected_date_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_date_columns,
    expected_identifier_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_identifier_fields,
    expected_categorical_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_categorical_fields,
    expected_approximate_row_count=4,
    large_file=False,
    notes='Small healthcare fixture for fast local and CI-friendly quick validation runs.',
)

SECONDARY_HEALTHCARE_FIXTURE = DatasetFixture(
    key='secondary-healthcare',
    path=FIXTURE_DATASET_DIR / 'ALT_EHP__VIST.csv',
    dataset_name='ALT_EHP__VIST.csv',
    dataset_type='healthcare',
    expected_key_source_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_key_source_columns,
    expected_date_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_date_columns,
    expected_identifier_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_identifier_fields,
    expected_categorical_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_categorical_fields,
    expected_approximate_row_count=5,
    large_file=False,
    notes='Second distinct healthcare upload fixture for cross-dataset cache contamination validation.',
)

MALFORMED_FIXTURE = DatasetFixture(
    key='malformed-healthcare',
    path=FIXTURE_DATASET_DIR / 'MALFORMED_VISITS.csv',
    dataset_name='MALFORMED_VISITS.csv',
    dataset_type='healthcare',
    expected_key_source_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_key_source_columns,
    expected_date_columns=PRIMARY_FULL_VALIDATION_FIXTURE.expected_date_columns,
    expected_identifier_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_identifier_fields,
    expected_categorical_fields=PRIMARY_FULL_VALIDATION_FIXTURE.expected_categorical_fields,
    expected_approximate_row_count=0,
    large_file=False,
    notes='Malformed healthcare fixture used to assert safe explicit failures during upload and parsing.',
)

AMBIGUOUS_MAPPING_FIXTURE = DatasetFixture(
    key='ambiguous-mapping-healthcare',
    path=FIXTURE_DATASET_DIR / 'AMBIGUOUS_ENCOUNTER_VISITS.csv',
    dataset_name='AMBIGUOUS_ENCOUNTER_VISITS.csv',
    dataset_type='healthcare',
    expected_key_source_columns=(
        'MEMBER_KEY',
        'VISIT_START_RAW',
        'VISIT_END_RAW',
        'VISIT_TYPE_LABEL',
        'VISIT_STATUS_LABEL',
        'ROOM_CODE',
    ),
    expected_date_columns=('VISIT_START_RAW', 'VISIT_END_RAW'),
    expected_identifier_fields=('MEMBER_KEY', 'ROOM_CODE'),
    expected_categorical_fields=('VISIT_TYPE_LABEL', 'VISIT_STATUS_LABEL'),
    expected_approximate_row_count=3,
    large_file=False,
    notes='Ambiguous encounter-style fixture used for manual mapping profile and exact on-screen assertions.',
)


FIXTURE_REGISTRY: dict[str, DatasetFixture] = {
    PRIMARY_FULL_VALIDATION_FIXTURE.key: PRIMARY_FULL_VALIDATION_FIXTURE,
    PRIMARY_FULL_VALIDATION_FIXTURE.dataset_name.lower(): PRIMARY_FULL_VALIDATION_FIXTURE,
    SMALL_HEALTHCARE_FIXTURE.key: SMALL_HEALTHCARE_FIXTURE,
    SMALL_HEALTHCARE_FIXTURE.dataset_name.lower(): SMALL_HEALTHCARE_FIXTURE,
    SECONDARY_HEALTHCARE_FIXTURE.key: SECONDARY_HEALTHCARE_FIXTURE,
    SECONDARY_HEALTHCARE_FIXTURE.dataset_name.lower(): SECONDARY_HEALTHCARE_FIXTURE,
    MALFORMED_FIXTURE.key: MALFORMED_FIXTURE,
    MALFORMED_FIXTURE.dataset_name.lower(): MALFORMED_FIXTURE,
    AMBIGUOUS_MAPPING_FIXTURE.key: AMBIGUOUS_MAPPING_FIXTURE,
    AMBIGUOUS_MAPPING_FIXTURE.dataset_name.lower(): AMBIGUOUS_MAPPING_FIXTURE,
    'default': PRIMARY_FULL_VALIDATION_FIXTURE,
}


def get_default_fixture() -> DatasetFixture:
    return PRIMARY_FULL_VALIDATION_FIXTURE


def get_fixture(name: str | None = None) -> DatasetFixture:
    if not name:
        return get_default_fixture()
    key = name.lower()
    if key not in FIXTURE_REGISTRY:
        raise KeyError(f'Unknown dataset fixture: {name}')
    return FIXTURE_REGISTRY[key]
