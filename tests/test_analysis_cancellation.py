from __future__ import annotations

import sys
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import pandas as pd

def _noop(*args, **kwargs):
    return None


if 'streamlit' not in sys.modules:
    streamlit_stub = types.ModuleType('streamlit')
    streamlit_stub.__spec__ = ModuleSpec('streamlit', loader=None)
    streamlit_stub.session_state = {}

    streamlit_stub.success = _noop
    streamlit_stub.warning = _noop
    streamlit_stub.info = _noop
    streamlit_stub.subheader = _noop
    streamlit_stub.caption = _noop
    streamlit_stub.write = _noop
    streamlit_stub.markdown = _noop
    streamlit_stub.selectbox = lambda *args, **kwargs: None
    streamlit_stub.number_input = lambda *args, **kwargs: 1
    streamlit_stub.progress = _noop
    streamlit_stub.empty = _noop
    streamlit_stub.button = lambda *args, **kwargs: False
    streamlit_stub.sidebar = types.SimpleNamespace()
    streamlit_stub.set_page_config = _noop
    sys.modules['streamlit'] = streamlit_stub

if 'plotly' not in sys.modules:
    plotly_stub = types.ModuleType('plotly')
    plotly_stub.__spec__ = ModuleSpec('plotly', loader=None)
    sys.modules['plotly'] = plotly_stub

if 'plotly.express' not in sys.modules:
    plotly_express_stub = types.ModuleType('plotly.express')
    plotly_express_stub.__spec__ = ModuleSpec('plotly.express', loader=None)
    plotly_express_stub.bar = _noop
    sys.modules['plotly.express'] = plotly_express_stub

if 'plotly.graph_objects' not in sys.modules:
    plotly_graph_objects_stub = types.ModuleType('plotly.graph_objects')
    plotly_graph_objects_stub.__spec__ = ModuleSpec('plotly.graph_objects', loader=None)
    plotly_graph_objects_stub.Figure = object
    plotly_graph_objects_stub.layout = types.SimpleNamespace(Template=lambda: None)
    plotly_graph_objects_stub.Layout = lambda **kwargs: kwargs
    sys.modules['plotly.graph_objects'] = plotly_graph_objects_stub

if 'plotly.io' not in sys.modules:
    plotly_io_stub = types.ModuleType('plotly.io')
    plotly_io_stub.__spec__ = ModuleSpec('plotly.io', loader=None)
    plotly_io_stub.templates = {}
    sys.modules['plotly.io'] = plotly_io_stub

import app


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.success_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.info_messages: list[str] = []

    def success(self, message: str) -> None:
        self.success_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)

    def info(self, message: str) -> None:
        self.info_messages.append(message)


class AnalysisCancellationTests(unittest.TestCase):
    def test_update_cancellation_countdown_switches_to_force_kill_message(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.session_state['analysis_cancellation'] = {
            'job-1': {
                'requested_at': 100.0,
                'timeout_seconds': 30,
                'status': 'pending',
            }
        }
        with patch.object(app, 'st', fake_st), patch.object(app.time, 'monotonic', return_value=115.2):
            snapshot = app._update_cancellation_countdown(
                'job-1',
                {'progress': 0.4, 'step_index': 3, 'total_steps': 5},
                source_meta={'file_size_mb': 65.8},
                row_count=917000,
                column_count=21,
            )

        self.assertEqual(snapshot['message'], 'Cancellation in progress (15 seconds remaining)')
        self.assertEqual(snapshot['current_operation'], 'Force-killing job (timeout in 15s)')
        self.assertTrue(snapshot['cancel_requested'])

    def test_force_cancel_managed_analysis_cleans_up_runtime_state(self) -> None:
        fake_st = _FakeStreamlit()
        fake_st.session_state.update(
            {
                'active_analysis_job': {'job_id': 'job-1', 'dataset_name': 'large.csv'},
                'analysis_progress': {'job-1': {'message': 'Running'}},
                'analysis_cancellation': {'job-1': {'requested_at': 10.0, 'timeout_seconds': 30}},
                'profile_cache_metrics': {'requests': 5, 'hits': 3},
                'generated_report_outputs': {'Executive Report': b'data'},
                'latest_dataset_artifact': {'artifact_path': 'artifact'},
                'active_dataset_bundle': {'data': pd.DataFrame([{'patient_id': 1}])},
            }
        )

        with patch.object(app, 'st', fake_st), patch.object(app, 'force_cancel_background_task', return_value={'status': 'cancelled'}):
            app._force_cancel_managed_analysis(
                'job-1',
                reason='Cancellation timed out after 30 seconds and the analysis job was force-killed.',
            )
            app._render_analysis_status_notice()

        self.assertNotIn('active_analysis_job', fake_st.session_state)
        self.assertNotIn('job-1', fake_st.session_state.get('analysis_progress', {}))
        self.assertNotIn('job-1', fake_st.session_state.get('analysis_cancellation', {}))
        self.assertEqual(fake_st.session_state['generated_report_outputs'], {})
        self.assertNotIn('latest_dataset_artifact', fake_st.session_state)
        self.assertIn('job-1', fake_st.session_state.get('ignored_analysis_jobs', set()))
        self.assertTrue(fake_st.success_messages)
        self.assertIn('Cancellation completed', fake_st.success_messages[0])


if __name__ == '__main__':
    unittest.main()
