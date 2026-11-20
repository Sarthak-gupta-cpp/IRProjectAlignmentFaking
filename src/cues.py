"""
Cue loading and formatting utilities.
Reads cue templates from configs/cues.yaml and applies them to prompts.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "cues.yaml"


def load_cues(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads all cue families and templates from YAML configuration."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Cue config not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("families", {})


def get_template(
    cue_family: str,
    template_id: str,
    families: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves a specific cue template by family and ID."""
    if families is None:
        families = load_cues()

    fam = families.get(cue_family)
    if not fam:
        raise ValueError(f"Unknown cue family: {cue_family}")

    # Search in implant and evaluation sets
    for split_type in ("implant", "evaluation"):
        for tmpl in fam.get(split_type, []):
            if tmpl.get("id") == template_id:
                return {
                    "family": cue_family,
                    "split_type": split_type,
                    "id": template_id,
                    "monitored": tmpl["monitored"],
                    "unmonitored": tmpl["unmonitored"],
                }

    raise ValueError(f"Template ID {template_id} not found in family {cue_family}")


def format_messages_with_cue(
    base_user_content: str,
    cue_text: str,
    base_system_prompt: Optional[str] = None,
    cue_as_system_prefix: bool = True,
) -> List[Dict[str, str]]:
    """
    Constructs chat messages (system and user) integrating the oversight cue.
    """
    messages = []
    if cue_as_system_prefix:
        sys_content = f"{cue_text}\n\n{base_system_prompt}" if base_system_prompt else cue_text
        messages.append({"role": "system", "content": sys_content.strip()})
        messages.append({"role": "user", "content": base_user_content})
    else:
        if base_system_prompt:
            messages.append({"role": "system", "content": base_system_prompt})
        user_content = f"{cue_text}\n\n{base_user_content}"
        messages.append({"role": "user", "content": user_content})

    return messages

