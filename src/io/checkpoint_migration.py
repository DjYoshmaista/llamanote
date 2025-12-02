"""
Checkpoint Migration Tool - Convert legacy checkpoints to new format.

Handles migration of checkpoints from old naming schema to new descriptive format,
including metadata generation and registry updates.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointMigrationTool:
    """
    Tool for migrating legacy checkpoints to new naming format.

    The migration process:
    1. Load legacy checkpoint and extract metadata
    2. Calculate completion percentage from stage/chunk info
    3. Generate new filename with model abbreviations
    4. Create .meta.json sidecar file
    5. Copy or rename checkpoint to new name
    6. Update registry with new entry
    7. Optionally mark old entry as 'legacy' or delete
    """

    def __init__(self, checkpoint_manager):
        """
        Initialize the migration tool.

        Args:
            checkpoint_manager: CheckpointManager instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

    def migrate_checkpoint(
        self,
        legacy_path: Path,
        keep_original: bool = True
    ) -> Optional[Path]:
        """
        Migrate a single legacy checkpoint to new format.

        Args:
            legacy_path: Path to legacy checkpoint file
            keep_original: If True, copy to new name. If False, rename (destructive)

        Returns:
            Path to new checkpoint file if successful, None otherwise
        """
        try:
            self.logger.info(f"Migrating checkpoint: {legacy_path.name}")

            # Load checkpoint to extract metadata
            result = self.checkpoint_manager.load(legacy_path)
            if result is None:
                self.logger.error(f"Failed to load legacy checkpoint: {legacy_path.name}")
                return None

            metadata, data = result

            # Extract information for new filename
            input_file_str = metadata.get("input_file", "")
            if not input_file_str:
                self.logger.error(f"No input file in metadata: {legacy_path.name}")
                return None

            input_path = Path(input_file_str)
            stage = metadata.get("stage", "unknown")
            chunk_index = metadata.get("chunk_index")
            total_chunks = metadata.get("total_chunks")

            # Calculate completion percentage
            completion_pct = None
            if chunk_index is not None and total_chunks is not None:
                try:
                    from ..config.settings import DEFAULT_STAGE_WEIGHTS
                    completed_stages = data.get('completed_stages', [])
                    completion_pct = self.checkpoint_manager.calculate_pipeline_completion(
                        current_stage=stage,
                        current_chunk=chunk_index,
                        total_chunks=total_chunks,
                        stage_weights=DEFAULT_STAGE_WEIGHTS,
                        completed_stages=completed_stages
                    )
                except Exception as e:
                    self.logger.warning(f"Could not calculate completion percentage: {e}")

            # Generate new filename using CheckpointManager logic
            # We need to reconstruct the config to generate proper filename
            from ..core.types import PipelineConfig

            # Extract config from metadata
            config_dict = metadata.get("full_config", {})
            if not config_dict or "_serialization_error" in config_dict:
                # Fallback: create minimal config from available metadata
                config_dict = self._create_minimal_config(metadata)

            try:
                config = PipelineConfig(**config_dict)
            except Exception as e:
                self.logger.warning(f"Could not reconstruct config: {e}")
                config = self._create_fallback_config(metadata)

            # Generate new checkpoint path
            new_path = self.checkpoint_manager._get_checkpoint_path(
                input_path=input_path,
                config=config,
                stage=stage,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                completion_pct=completion_pct
            )

            # Check if new path already exists
            if new_path.exists():
                self.logger.warning(f"New checkpoint already exists: {new_path.name}")
                return new_path

            # Copy or rename checkpoint
            if keep_original:
                import shutil
                shutil.copy2(legacy_path, new_path)
                self.logger.info(f"Copied to: {new_path.name}")
            else:
                legacy_path.rename(new_path)
                self.logger.info(f"Renamed to: {new_path.name}")

            # Create metadata file
            compressed_size = new_path.stat().st_size
            uncompressed_size = compressed_size if not self.checkpoint_manager.enable_compression else int(compressed_size * 1.7)

            self.checkpoint_manager._create_metadata_file(
                checkpoint_path=new_path,
                metadata=metadata,
                checkpoint_size_bytes=uncompressed_size,
                compressed_size_bytes=compressed_size,
                checkpoint_hash=None
            )

            # Update registry
            if self.checkpoint_manager.registry is not None:
                file_size_mb = round(compressed_size / (1024 * 1024), 2)
                self.checkpoint_manager._update_registry_entry(
                    checkpoint_path=new_path,
                    metadata=metadata,
                    input_path=input_path,
                    compressed_size_mb=file_size_mb
                )

            # If we kept the original, mark it as legacy in registry
            if keep_original and self.checkpoint_manager.registry is not None:
                self.checkpoint_manager.registry.update_entry(
                    legacy_path.name,
                    {"status": "legacy"}
                )

            self.logger.info(f"Migration successful: {legacy_path.name} -> {new_path.name}")
            return new_path

        except Exception as e:
            self.logger.error(f"Migration failed for {legacy_path.name}: {e}", exc_info=True)
            return None

    def migrate_all_legacy(
        self,
        keep_originals: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Migrate all legacy checkpoints to new format.

        Args:
            keep_originals: If True, keep legacy checkpoints. If False, delete after migration.
            progress_callback: Optional callback(current, total, checkpoint_name)

        Returns:
            Dictionary with migration statistics
        """
        stats = {
            "total_found": 0,
            "success_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "errors": []
        }

        # Find all legacy checkpoints
        legacy_checkpoints = self.checkpoint_manager.find_legacy_checkpoints()
        stats["total_found"] = len(legacy_checkpoints)

        if not legacy_checkpoints:
            self.logger.info("No legacy checkpoints found")
            return stats

        self.logger.info(f"Found {len(legacy_checkpoints)} legacy checkpoint(s)")

        for idx, legacy_path in enumerate(legacy_checkpoints, 1):
            try:
                new_path = self.migrate_checkpoint(legacy_path, keep_original=keep_originals)

                if new_path:
                    stats["success_count"] += 1
                else:
                    stats["error_count"] += 1
                    stats["errors"].append(f"Failed to migrate: {legacy_path.name}")

                # Call progress callback
                if progress_callback:
                    progress_callback(idx, stats["total_found"], legacy_path.name)

            except Exception as e:
                error_msg = f"Error migrating {legacy_path.name}: {str(e)}"
                self.logger.error(error_msg)
                stats["error_count"] += 1
                stats["errors"].append(error_msg)

        # Log summary
        self.logger.info(
            f"Migration complete: "
            f"{stats['success_count']} migrated, "
            f"{stats['error_count']} errors"
        )

        return stats

    def _create_minimal_config(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create minimal config dict from metadata.

        Args:
            metadata: Checkpoint metadata

        Returns:
            Minimal config dictionary
        """
        config = metadata.get("config", {})

        # Extract text model
        text_model = config.get("text_model", {})
        if isinstance(text_model, dict):
            model_provider = text_model.get("provider", "unknown")
            model_specifier = text_model.get("specifier", "unknown")
        else:
            parts = str(text_model).split(":", 1)
            model_provider = parts[0] if len(parts) > 1 else "unknown"
            model_specifier = parts[1] if len(parts) > 1 else str(text_model)

        # Extract audio model if available
        audio_config = config.get("audio", {})
        audio_provider = audio_config.get("provider", "unknown")
        audio_specifier = audio_config.get("model", "unknown")

        return {
            "model_provider": model_provider,
            "model_specifier": model_specifier,
            "audio_provider": audio_provider,
            "audio_specifier": audio_specifier,
            "mode": str(config.get("mode", "podcast")),
            "generate_audio": bool(audio_config),
        }

    def _create_fallback_config(self, metadata: Dict[str, Any]):
        """
        Create fallback PipelineConfig when reconstruction fails.

        Args:
            metadata: Checkpoint metadata

        Returns:
            Minimal PipelineConfig instance
        """
        from ..core.types import PipelineConfig, ProcessingMode

        minimal = self._create_minimal_config(metadata)

        return PipelineConfig(
            model_provider=minimal.get("model_provider", "unknown"),
            model_specifier=minimal.get("model_specifier", "unknown"),
            mode=ProcessingMode.PODCAST,  # Default
            generate_audio=minimal.get("generate_audio", False)
        )
