# src/heylook_llm/config_migration.py
"""Utility for migrating YAML configuration to TOML format."""

import yaml
import tomli_w
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def migrate_yaml_to_toml(
    yaml_path: str,
    toml_path: Optional[str] = None,
    backup: bool = True
) -> Path:
    """Migrate a YAML configuration file to TOML format.

    Args:
        yaml_path: Path to the source YAML file
        toml_path: Path to the output TOML file (default: same name with .toml extension)
        backup: Whether to create a backup of the original YAML file

    Returns:
        Path to the created TOML file

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the TOML file already exists and backup is False
    """
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    # Determine output path
    if toml_path is None:
        toml_file = yaml_file.with_suffix('.toml')
    else:
        toml_file = Path(toml_path)

    # Check if TOML already exists
    if toml_file.exists():
        logging.warning(f"TOML file already exists: {toml_file}")
        if not backup:
            raise ValueError(
                f"TOML file already exists: {toml_file}. "
                f"Set backup=True to overwrite or specify a different toml_path."
            )

    # Load YAML
    logging.info(f"Loading YAML configuration from {yaml_file}")
    with open(yaml_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # Write TOML
    logging.info(f"Writing TOML configuration to {toml_file}")
    with open(toml_file, 'wb') as f:
        tomli_w.dump(config_data, f)

    # Backup original YAML
    if backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = yaml_file.with_suffix(f'.yaml.backup.{timestamp}')
        logging.info(f"Creating backup: {backup_file}")
        shutil.copy2(yaml_file, backup_file)

    logging.info(f"Migration complete: {yaml_file} → {toml_file}")
    if backup:
        logging.info(f"Original YAML backed up to: {backup_file}")

    return toml_file


def migrate_models_config(
    base_dir: str = ".",
    backup: bool = True
) -> Optional[Path]:
    """Migrate models.yaml to models.toml in a directory.

    Convenience function that looks for models.yaml and migrates it to models.toml.

    Args:
        base_dir: Directory containing models.yaml
        backup: Whether to create a backup of the original YAML file

    Returns:
        Path to the created models.toml file, or None if models.yaml doesn't exist
    """
    base_path = Path(base_dir)
    yaml_file = base_path / "models.yaml"
    toml_file = base_path / "models.toml"

    if not yaml_file.exists():
        logging.info(f"No models.yaml found in {base_dir}")
        return None

    if toml_file.exists():
        logging.info(f"models.toml already exists in {base_dir}, skipping migration")
        return toml_file

    return migrate_yaml_to_toml(str(yaml_file), str(toml_file), backup=backup)


if __name__ == "__main__":
    # CLI usage: python -m heylook_llm.config_migration [yaml_path] [toml_path]
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python -m heylook_llm.config_migration <yaml_path> [toml_path]")
        print("   or: python -m heylook_llm.config_migration --migrate-models [base_dir]")
        sys.exit(1)

    if sys.argv[1] == "--migrate-models":
        base_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        result = migrate_models_config(base_dir)
        if result:
            print(f"✓ Migration complete: {result}")
        else:
            print("✗ No models.yaml found to migrate")
    else:
        yaml_path = sys.argv[1]
        toml_path = sys.argv[2] if len(sys.argv) > 2 else None
        result = migrate_yaml_to_toml(yaml_path, toml_path)
        print(f"✓ Migration complete: {result}")
