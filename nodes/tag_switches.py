from __future__ import annotations

from pathlib import Path

from ..services.tag_index import TagDataError, load_tag_index, tag_switch_group


_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tags.csv"


def _switch_inputs(group: str) -> dict[str, str]:
    index = load_tag_index(_DATA_PATH)
    inputs: dict[str, str] = {}
    for switch in sorted(index.tag_switches):
        if tag_switch_group(switch) != group:
            continue
        name = switch.rsplit(".", 1)[-1]
        if name in inputs:
            raise TagDataError(f"Duplicate switch input {name!r} in group {group!r}")
        inputs[name] = switch
    return dict(sorted(inputs.items()))


class _TagSwitchNode:
    CATEGORY = "Anima Prompt/Tag Switches"
    FUNCTION = "build"
    RETURN_TYPES = ("ANIMA_TAG_SWITCH_LIST",)
    RETURN_NAMES = ("switch_list",)
    SWITCH_GROUP = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                name: ("BOOLEAN", {"default": True})
                for name in _switch_inputs(cls.SWITCH_GROUP)
            },
            "optional": {
                "switch_list": ("ANIMA_TAG_SWITCH_LIST",),
            },
        }

    def build(self, switch_list=None, **values):
        inputs = _switch_inputs(self.SWITCH_GROUP)
        prefix = f"{self.SWITCH_GROUP}."
        enabled = {
            switch for switch in (switch_list or ()) if not switch.startswith(prefix)
        }
        enabled.update(switch for name, switch in inputs.items() if values[name])
        return (frozenset(enabled),)


class VisualCompositionSwitches(_TagSwitchNode):
    SWITCH_GROUP = "visual_composition"


class SubjectAppearanceSwitches(_TagSwitchNode):
    SWITCH_GROUP = "subject_appearance"


class LivingNatureSwitches(_TagSwitchNode):
    SWITCH_GROUP = "living_nature"


class ScenesActivitiesCultureSwitches(_TagSwitchNode):
    SWITCH_GROUP = "scenes_activities_culture"


class ObjectsEquipmentSwitches(_TagSwitchNode):
    SWITCH_GROUP = "objects_equipment"


class AdultContentSwitches(_TagSwitchNode):
    SWITCH_GROUP = "adult_content"
