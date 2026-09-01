"""
Path Configuration Module for Gorilla Tag Project
=================================================

This module provides standardized path configuration across all notebooks and scripts
in the Gorilla Tag project. It automatically detects the host computer and sets
appropriate paths for data directories.

Usage:
    from config_paths import get_project_paths

    paths = get_project_paths()
    print(f"Data directory: {paths.data_dir}")
    print(f"Raw data directory: {paths.raw_data_dir}")
    print(f"Repetitive data directory: {paths.repetitive_data_dir}")
    print(f"Repetitive raw data directory: {paths.repetitive_raw_data_dir}")
    print(f"Processed data directory: {paths.processed_data_dir}")
    print(f"Image directory: {paths.img_dir}")
    print(f"BVH directory: {paths.bvh_dir}")
    print(f"Hostname: {paths.hostname}")

"""

import os
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectPaths:
    """Container for all project paths."""

    root_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    img_dir: Path
    bvh_dir: Path
    hostname: str
    repetitive_data_dir: Path
    repetitive_raw_data_dir: Path  
    repetitive_processed_data_dir: Path
    repetitive_bvh_dir: Path

    def __post_init__(self):
        """Ensure all paths exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.img_dir,
            self.bvh_dir,
            self.repetitive_data_dir,
            self.repetitive_raw_data_dir,  
            self.repetitive_processed_data_dir,
            self.repetitive_bvh_dir
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_project_paths(
    hostname: Optional[str] = None,
    root_dir: Optional[Path] = None,
) -> ProjectPaths:
    """
    Get project paths based on hostname detection.

    Args:
        hostname: Optional hostname override. If None, auto-detects using platform.node().
        root_dir: Optional explicit project root. When omitted, the
            ``GORILLA_TAG_PROJECT_ROOT`` environment variable is honored before
            the legacy hostname mapping.

    Returns:
        ProjectPaths: Dataclass containing all project paths
    """
    if hostname is None:
        hostname = platform.node()

    print(f"Detected hostname: {hostname}")

    if root_dir is None:
        environment_root = os.environ.get("GORILLA_TAG_PROJECT_ROOT")
        if environment_root:
            root_dir = Path(environment_root).expanduser().resolve()
        elif hostname == "Opher-Lab-8a":  # Desktop
            root_dir = Path(r"D:\noamgonen\GitHub\gorilla_tag_")
        elif hostname == "DESKTOP-P681IHN":  # Opher's Lenovo
            root_dir = Path(r"C:\Repositories\gorilla_tag_")
        elif hostname == "NOAM-LAPTOP":  # Noam's Laptop
            root_dir = Path(r"C:\GitHub\gorilla_tag_")
        else:
            raise ValueError("Unknown hostname and no explicit root directory provided." \
            " please set the GORILLA_TAG_PROJECT_ROOT environment variable or provide a root_dir argument.")
    else:
        root_dir = Path(root_dir).expanduser().resolve()

    print(f"Using root directory: {root_dir}")

    # Construct all paths
    # "free practice" group data"
    data_dir = root_dir / "data"
    raw_data_dir = data_dir / "raw_data"
    processed_data_dir = data_dir / "processed_data"
    img_dir = root_dir / "img"
    bvh_dir = data_dir / "raw_bvh"
    # "repetitive data" group data
    repetitive_data_dir = root_dir / "repetitive_data"
    repetitive_raw_data_dir = repetitive_data_dir / "raw_data"
    repetitive_processed_data_dir = repetitive_data_dir / "processed_data"
    repetitive_bvh_dir = repetitive_data_dir / "raw_bvh"    



    return ProjectPaths(
        root_dir=root_dir,
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        img_dir=img_dir,
        bvh_dir=bvh_dir,
        hostname=hostname,
        repetitive_data_dir=repetitive_data_dir,
        repetitive_raw_data_dir=repetitive_raw_data_dir,
        repetitive_processed_data_dir=repetitive_processed_data_dir,
        repetitive_bvh_dir=repetitive_bvh_dir
    )


# Global paths instance for convenience (initialized on import)
PATHS = get_project_paths()

# Export commonly used paths as module-level variables for easy access
img_dir = PATHS.img_dir
csv_dir = PATHS.raw_data_dir
aligned_csv_dir = PATHS.processed_data_dir 
bvh_dir = PATHS.bvh_dir


if __name__ == "__main__":
    # Test the configuration
    paths = get_project_paths()
    print("\n=== Project Path Configuration ===")
    print(f"Root directory: {paths.root_dir}")
    print(f"Raw CSV data directory: {paths.raw_data_dir}")
    print(f"Aligned CSV data directory: {paths.processed_data_dir}")
    print(f"Image directory: {paths.img_dir}")
    print(f"BVH directory: {paths.bvh_dir}")
    print(f"Hostname: {paths.hostname}")
    print(f"Repetitive data directory: {paths.repetitive_data_dir}")
    print(f"Repetitive raw csv data directory: {paths.repetitive_raw_data_dir}")
    print(f"Repetitive aligned csv data directory: {paths.repetitive_processed_data_dir}")
    print(f"Repetitive BVH directory: {paths.repetitive_bvh_dir}")

    print("\n=== Path Existence Check ===")
    for name, path in [
        ("Root", paths.root_dir),
        ("Data", paths.data_dir),
        ("Raw CSV Data", paths.raw_data_dir),
        ("Aligned CSV Data", paths.processed_data_dir),
        ("Images", paths.img_dir),
        ("BVH", paths.bvh_dir),
        ("Repetitive Data", paths.repetitive_data_dir),
        ("Repetitive Raw CSV Data", paths.repetitive_raw_data_dir),
        ("Repetitive Aligned CSV Data", paths.repetitive_processed_data_dir),
        ("Repetitive BVH", paths.repetitive_bvh_dir)
    ]:
        exists = "✓" if path.exists() else "✗"
        print(f"{name:15} {exists} {path}")
