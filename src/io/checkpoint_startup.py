"""
Checkpoint Startup Helper - Automatic maintenance and verification on startup.

Provides user prompts and automated tasks for:
- Metadata file generation for checkpoints
- Legacy checkpoint migration
- Registry verification and synchronization
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointStartupHelper:
    """
    Helper for checkpoint maintenance tasks on application startup.

    Handles:
    1. Registry verification and synchronization
    2. Missing metadata detection and generation
    3. Legacy checkpoint detection and migration
    4. User prompts for maintenance actions
    """

    def __init__(self, checkpoint_manager):
        """
        Initialize the startup helper.

        Args:
            checkpoint_manager: CheckpointManager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

    def run_startup_checks(
        self,
        auto_sync_registry: bool = True,
        prompt_metadata_generation: bool = True,
        prompt_legacy_migration: bool = True,
        interactive: bool = True
    ) -> Dict[str, Any]:
        """
        Run all startup checks and maintenance tasks.

        Args:
            auto_sync_registry: Automatically verify and sync registry
            prompt_metadata_generation: Prompt user to generate missing metadata
            prompt_legacy_migration: Prompt user to migrate legacy checkpoints
            interactive: If False, skip prompts and use defaults

        Returns:
            Dictionary with results from all checks
        """
        results = {
            "registry_sync": None,
            "metadata_check": None,
            "legacy_check": None,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info("Running checkpoint startup checks...")

        # 1. Verify and sync registry
        if auto_sync_registry:
            self.logger.info("Verifying checkpoint registry...")
            sync_result = self._verify_registry()
            results["registry_sync"] = sync_result
            self._report_registry_sync(sync_result)

        # 2. Check for missing metadata
        if prompt_metadata_generation:
            self.logger.info("Checking for missing metadata files...")
            metadata_result = self._check_missing_metadata(interactive)
            results["metadata_check"] = metadata_result

        # 3. Check for legacy checkpoints
        if prompt_legacy_migration:
            self.logger.info("Checking for legacy checkpoints...")
            legacy_result = self._check_legacy_checkpoints(interactive)
            results["legacy_check"] = legacy_result

        self.logger.info("Startup checks complete")
        return results

    def _verify_registry(self) -> Dict[str, Any]:
        """
        Verify and synchronize the checkpoint registry.

        Returns:
            Dictionary with verification results
        """
        try:
            result = self.checkpoint_manager.verify_and_sync_registry()
            return {
                "success": True,
                "removed_stale": result["removed_stale"],
                "found_new": result["found_new"],
                "missing_metadata": result["missing_metadata"],
                "error": None
            }
        except Exception as e:
            self.logger.error(f"Registry verification failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _check_missing_metadata(self, interactive: bool = True) -> Dict[str, Any]:
        """
        Check for checkpoints without metadata and prompt generation.

        Args:
            interactive: If True, prompt user. If False, auto-generate.

        Returns:
            Dictionary with check results
        """
        try:
            missing = self.checkpoint_manager.find_checkpoints_without_metadata()
            count = len(missing)

            if count == 0:
                self.logger.info("All checkpoints have metadata files")
                return {
                    "missing_count": 0,
                    "generated": 0,
                    "action": "none_needed"
                }

            self.logger.info(f"Found {count} checkpoint(s) without metadata")

            # Decide whether to generate
            should_generate = False
            if interactive:
                should_generate = self._prompt_metadata_generation(count)
            else:
                should_generate = True  # Auto-generate in non-interactive mode

            if should_generate:
                self.logger.info(f"Generating metadata for {count} checkpoint(s)...")
                stats = self.checkpoint_manager.regenerate_metadata_files(
                    checkpoint_paths=missing,
                    progress_callback=self._metadata_progress_callback
                )

                return {
                    "missing_count": count,
                    "generated": stats["success_count"],
                    "errors": stats["error_count"],
                    "action": "generated"
                }
            else:
                self.logger.info("Metadata generation deferred")
                return {
                    "missing_count": count,
                    "generated": 0,
                    "action": "deferred"
                }

        except Exception as e:
            self.logger.error(f"Metadata check failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "action": "error"
            }

    def _check_legacy_checkpoints(self, interactive: bool = True) -> Dict[str, Any]:
        """
        Check for legacy checkpoints and prompt migration.

        Args:
            interactive: If True, prompt user. If False, use defaults.

        Returns:
            Dictionary with check results
        """
        try:
            legacy_checkpoints = self.checkpoint_manager.find_legacy_checkpoints()
            count = len(legacy_checkpoints)

            if count == 0:
                self.logger.info("No legacy checkpoints found")
                return {
                    "legacy_count": 0,
                    "migrated": 0,
                    "action": "none_needed"
                }

            self.logger.info(f"Found {count} legacy checkpoint(s)")

            # Decide whether to migrate
            should_migrate = False
            keep_originals = True

            if interactive:
                should_migrate, keep_originals = self._prompt_legacy_migration(count)
            else:
                should_migrate = False  # Don't auto-migrate in non-interactive mode

            if should_migrate:
                from .checkpoint_migration import CheckpointMigrationTool
                migration_tool = CheckpointMigrationTool(self.checkpoint_manager)

                self.logger.info(f"Migrating {count} legacy checkpoint(s)...")
                stats = migration_tool.migrate_all_legacy(
                    keep_originals=keep_originals,
                    progress_callback=self._migration_progress_callback
                )

                return {
                    "legacy_count": count,
                    "migrated": stats["success_count"],
                    "errors": stats["error_count"],
                    "kept_originals": keep_originals,
                    "action": "migrated"
                }
            else:
                self.logger.info("Legacy migration deferred")
                return {
                    "legacy_count": count,
                    "migrated": 0,
                    "action": "deferred"
                }

        except Exception as e:
            self.logger.error(f"Legacy checkpoint check failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "action": "error"
            }

    def _prompt_metadata_generation(self, count: int) -> bool:
        """
        Prompt user to generate metadata files.

        Args:
            count: Number of checkpoints without metadata

        Returns:
            True if user wants to generate, False otherwise
        """
        print("\n" + "=" * 70)
        print(f"Checkpoint Metadata Check")
        print("=" * 70)
        print(f"Found {count} checkpoint(s) without metadata files.")
        print("\nMetadata files enable:")
        print("  - Faster checkpoint discovery")
        print("  - Human-readable checkpoint information")
        print("  - Registry-based filtering and search")
        print("\nGenerate metadata files now?")
        print("  [y] Yes - Generate metadata now")
        print("  [n] No - Skip for now (can generate later from menu)")
        print("=" * 70)

        while True:
            try:
                response = input("Choice [y/n]: ").strip().lower()
                if response in ('y', 'yes'):
                    return True
                elif response in ('n', 'no'):
                    return False
                else:
                    print("Please enter 'y' or 'n'")
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping metadata generation")
                return False

    def _prompt_legacy_migration(self, count: int) -> Tuple[bool, bool]:
        """
        Prompt user to migrate legacy checkpoints.

        Args:
            count: Number of legacy checkpoints

        Returns:
            Tuple of (should_migrate, keep_originals)
        """
        print("\n" + "=" * 70)
        print(f"Legacy Checkpoint Detection")
        print("=" * 70)
        print(f"Found {count} checkpoint(s) using old naming format.")
        print("\nOld format: {hash}_{stage}_chunk{N}_{timestamp}.ckpt")
        print("New format: {file}_{ext}-{model1}-{model2}-chk{N}-{pct}pct.ckpt")
        print("\nBenefits of new format:")
        print("  - Human-readable filenames")
        print("  - Shows progress at a glance")
        print("  - Better sorting and organization")
        print("\nMigrate legacy checkpoints now?")
        print("  [y] Yes - Migrate and keep originals (safe)")
        print("  [d] Yes - Migrate and delete originals (saves space)")
        print("  [n] No - Skip for now (can migrate later from menu)")
        print("=" * 70)

        while True:
            try:
                response = input("Choice [y/d/n]: ").strip().lower()
                if response in ('y', 'yes'):
                    return (True, True)  # Migrate, keep originals
                elif response in ('d', 'delete'):
                    return (True, False)  # Migrate, delete originals
                elif response in ('n', 'no'):
                    return (False, True)  # Don't migrate
                else:
                    print("Please enter 'y', 'd', or 'n'")
            except (EOFError, KeyboardInterrupt):
                print("\nSkipping migration")
                return (False, True)

    def _metadata_progress_callback(self, current: int, total: int, checkpoint_name: str):
        """Progress callback for metadata generation."""
        print(f"  [{current}/{total}] Generating metadata for: {checkpoint_name}")

    def _migration_progress_callback(self, current: int, total: int, checkpoint_name: str):
        """Progress callback for migration."""
        print(f"  [{current}/{total}] Migrating: {checkpoint_name}")

    def _report_registry_sync(self, result: Dict[str, Any]):
        """
        Print registry synchronization results.

        Args:
            result: Sync result dictionary
        """
        if not result["success"]:
            self.logger.error(f"Registry sync failed: {result.get('error', 'Unknown error')}")
            return

        removed = result["removed_stale"]
        found_new = result["found_new"]
        missing_meta = result["missing_metadata"]

        if removed == 0 and found_new == 0 and missing_meta == 0:
            self.logger.info("Registry is up to date")
            return

        print("\n" + "=" * 70)
        print("Checkpoint Registry Synchronization")
        print("=" * 70)

        if removed > 0:
            print(f"  Removed {removed} stale entry/entries (checkpoint files deleted)")

        if found_new > 0:
            print(f"  Found {found_new} new checkpoint(s) not in registry")

        if missing_meta > 0:
            print(f"  Found {missing_meta} checkpoint(s) without metadata files")

        print("=" * 70 + "\n")


def run_startup_checks_default(checkpoint_manager) -> Dict[str, Any]:
    """
    Convenience function to run default startup checks.

    Args:
        checkpoint_manager: CheckpointManager instance

    Returns:
        Dictionary with check results
    """
    helper = CheckpointStartupHelper(checkpoint_manager)
    return helper.run_startup_checks(
        auto_sync_registry=True,
        prompt_metadata_generation=True,
        prompt_legacy_migration=True,
        interactive=True
    )
