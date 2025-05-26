# src/oak/knowledge_base/kb_loader.py
"""
Logic for loading and managing the Knowledge Base from the file system.
"""

import json
from pathlib import Path
from typing import Dict, List
import warnings # Added for more formal warnings

from .hardware_profile import HardwareProfile

class KnowledgeBaseError(Exception):
    """Base exception for errors related to the Knowledge Base."""
    pass

class KnowledgeBase:
    """
    Manages access to hardware profiles and (eventually) benchmark results.
    """
    def __init__(self, kb_path: Path):
        """
        Initializes the KnowledgeBase.

        Args:
            kb_path: Path to the root directory of the knowledge base (e.g., 'data/').

        Raises:
            KnowledgeBaseError: If kb_path is not a valid directory.
        """
        if not kb_path.is_dir():
            raise KnowledgeBaseError(f"Knowledge Base path is not a directory: {kb_path}")
        self.hardware_profiles_path = kb_path / "hardware"
        self.hardware: Dict[str, HardwareProfile] = self._load_hardware_profiles()

    def _load_hardware_profiles(self) -> Dict[str, HardwareProfile]:
        """
        Loads all hardware profiles from the 'data/hardware' directory.

        Returns:
            A dictionary mapping hardware identifiers to HardwareProfile objects.
        """
        profiles: Dict[str, HardwareProfile] = {}
        if not self.hardware_profiles_path.is_dir():
            # Allows the 'hardware' folder to not exist if there are no profiles yet.
            return profiles

        for file_path in self.hardware_profiles_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f: # Added encoding
                    data = json.load(f)
                    profile = HardwareProfile(**data)
                    if profile.identifier in profiles:
                        # This should ideally be a more specific error or logged,
                        # but for now, a KnowledgeBaseError is raised.
                        raise KnowledgeBaseError(
                            f"Duplicate hardware identifier found: {profile.identifier} in {file_path.name} "
                            f"(already loaded from another file)."
                        )
                    profiles[profile.identifier] = profile
            except json.JSONDecodeError as e:
                warnings.warn(
                    f"Warning: Failed to decode JSON for profile {file_path.name}. Error: {e}. Skipping this file.",
                    UserWarning
                )
            except Exception as e: # Catches Pydantic's ValidationError and other unexpected errors
                # Using warnings module for better warning handling.
                # In the future, a dedicated logging mechanism would be better.
                warnings.warn(
                    f"Warning: Failed to load or validate profile {file_path.name}. Error: {e}. Skipping this file.",
                    UserWarning
                )
        return profiles

    def list_hardware_identifiers(self) -> List[str]:
        """Returns a list of all loaded hardware identifiers."""
        return list(self.hardware.keys())

    def get_hardware(self, identifier: str) -> HardwareProfile:
        """
        Retrieves a hardware profile by its identifier.

        Args:
            identifier: The unique identifier of the hardware.

        Returns:
            The corresponding HardwareProfile object.

        Raises:
            KnowledgeBaseError: If the hardware profile is not found.
        """
        profile = self.hardware.get(identifier)
        if not profile:
            raise KnowledgeBaseError(f"Hardware profile '{identifier}' not found.")
        return profile