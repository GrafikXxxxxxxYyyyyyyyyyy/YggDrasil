"""Tests for serving param_utils (G3: extra params merge). No torch required."""
import importlib.util
from pathlib import Path

# Load param_utils without importing yggdrasil (avoids torch)
_p = Path(__file__).resolve().parent.parent / "yggdrasil" / "serving" / "param_utils.py"
_spec = importlib.util.spec_from_file_location("param_utils", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
merge_extra_params_json = _mod.merge_extra_params_json
infer_input_visibility = _mod.infer_input_visibility


class TestMergeExtraParamsJson:
    """merge_extra_params_json: merge extra JSON params into kwargs."""

    def test_empty_json_returns_unchanged(self):
        kwargs = {"prompt": "x", "num_steps": 28}
        assert merge_extra_params_json(kwargs, "") == kwargs
        assert merge_extra_params_json(kwargs, "{}") == kwargs
        assert merge_extra_params_json(kwargs, "   ") == kwargs

    def test_adds_missing_keys(self):
        kwargs = {"prompt": "a cat"}
        extra = '{"ip_adapter_scale": 0.5, "strength": 0.8}'
        result = merge_extra_params_json(kwargs, extra)
        assert result["prompt"] == "a cat"
        assert result["ip_adapter_scale"] == 0.5
        assert result["strength"] == 0.8

    def test_does_not_override_existing_values(self):
        kwargs = {"prompt": "set", "num_steps": 20}
        extra = '{"prompt": "override", "num_steps": 50, "new_key": 1}'
        result = merge_extra_params_json(kwargs, extra)
        assert result["prompt"] == "set"
        assert result["num_steps"] == 20
        assert result["new_key"] == 1

    def test_overrides_none_values(self):
        kwargs = {"prompt": "x", "ip_adapter_scale": None}
        extra = '{"ip_adapter_scale": 0.7}'
        result = merge_extra_params_json(kwargs, extra)
        assert result["ip_adapter_scale"] == 0.7

    def test_invalid_json_returns_unchanged(self):
        kwargs = {"prompt": "x"}
        assert merge_extra_params_json(kwargs, "{invalid") == kwargs
        assert merge_extra_params_json(kwargs, "not json") == kwargs

    def test_non_dict_json_returns_unchanged(self):
        kwargs = {"prompt": "x"}
        assert merge_extra_params_json(kwargs, "[1,2,3]") == kwargs
        assert merge_extra_params_json(kwargs, '"string"') == kwargs


class TestInferInputVisibility:
    """infer_input_visibility: (template_name, modality) -> (control, ip, source) visibility."""

    def test_controlnet_shows_control(self):
        ctrl, ip, src = infer_input_visibility("controlnet_sdxl_txt2img", "image")
        assert ctrl is True
        assert ip is True
        assert src is False

    def test_img2img_shows_source(self):
        ctrl, ip, src = infer_input_visibility("sd15_img2img", "image")
        assert src is True

    def test_plain_txt2img_hides_control_and_source(self):
        ctrl, ip, src = infer_input_visibility("sd15_txt2img", "image")
        assert ctrl is False
        assert ip is True
        assert src is False

    def test_video_shows_source(self):
        _, _, src = infer_input_visibility("animate_txt2vid", "video")
        assert src is True
