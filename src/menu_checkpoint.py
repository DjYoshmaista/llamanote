# llamanote/menu_checkpoint.py
"""
Checkpoint Management Menu Module
Handles checkpoint browsing, loading, viewing, and management operations.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

from .utils.logger import get_logger_conf, ConsoleOutput
from .io.checkpoints import CheckpointManager
from .core.types import PipelineConfig

logger = get_logger_conf(__name__)


class CheckpointMenuManager:
    """Manages checkpoint-related menu operations."""

    def __init__(self, checkpoint_base_dir: Optional[Path] = None):
        """
        Initialize checkpoint menu manager.

        Args:
            checkpoint_base_dir: Root checkpoint directory (defaults to project_root/checkpoints)
        """
        if checkpoint_base_dir is None:
            checkpoint_base_dir = Path(__file__).parent.parent / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_base_dir)
        self.checkpoint_manager = CheckpointManager(base_checkpoint_dir=self.checkpoint_dir)
        self.logger = logger

    def list_all_checkpoints(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        List all checkpoint files with their metadata.

        Returns:
            List of (checkpoint_path, metadata) tuples
        """
        checkpoints = []

        if not self.checkpoint_dir.exists():
            return checkpoints

        # Find all .ckpt files recursively
        for ckpt_file in self.checkpoint_dir.rglob("*.ckpt"):
            try:
                result = self.checkpoint_manager.load(ckpt_file)
                if result:
                    metadata, _ = result
                    checkpoints.append((ckpt_file, metadata))
            except Exception as e:
                # Silently skip corrupted checkpoints during listing
                # Full error details are logged in checkpoints.py
                self.logger.debug(f"Skipping corrupted checkpoint during listing: {ckpt_file.name}", exc_info=True)
                continue

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
        return checkpoints

    def list_checkpoints_by_input(self, input_file_stem: str) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        List checkpoints for a specific input file.

        Args:
            input_file_stem: Stem of the input file (without extension)

        Returns:
            List of (checkpoint_path, metadata) tuples
        """
        all_checkpoints = self.list_all_checkpoints()
        return [
            (path, meta) for path, meta in all_checkpoints
            if meta.get('input_stem') == input_file_stem
        ]

    def display_checkpoint_info(self, checkpoint_path: Path) -> bool:
        """
        Display detailed information about a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if displayed successfully
        """
        result = self.checkpoint_manager.load(checkpoint_path)
        if not result:
            ConsoleOutput.error(f"Failed to load checkpoint: {checkpoint_path.name}")
            return False

        metadata, data = result

        print("\n" + "=" * 80)
        print(f"{'Checkpoint Details':^80}")
        print("=" * 80)

        # Basic info
        print(f"\n📄 File: {checkpoint_path.name}")
        print(f"📁 Location: {checkpoint_path.parent}")
        print(f"📅 Created: {metadata.get('timestamp', 'Unknown')}")
        print(f"📊 Stage: {metadata.get('stage', 'Unknown')}")

        # Input file info
        print(f"\n📖 Input File:")
        print(f"   Name: {metadata.get('input_stem', 'Unknown')}")
        print(f"   Path: {metadata.get('input_file', 'Unknown')}")

        # Configuration
        config = metadata.get('config', {})
        print(f"\n⚙️  Configuration:")
        print(f"   Text Model: {config.get('text_model', 'N/A')}")
        print(f"   Mode: {config.get('mode', 'N/A')}")
        print(f"   Output Format: {config.get('output_format', 'N/A')}")

        # Chunking info
        chunking = config.get('chunking')
        if chunking:
            print(f"\n📦 Chunking:")
            print(f"   Strategy: {chunking.get('strategy', 'N/A')}")
            print(f"   Chunk Size: {chunking.get('chunk_size', 'N/A')}")
            print(f"   Overlap: {chunking.get('chunk_overlap', 'N/A')}")

        # Audio info
        audio_cfg = config.get('audio')
        if audio_cfg:
            print(f"\n🔊 Audio:")
            print(f"   Provider: {audio_cfg.get('provider', 'N/A')}")
            print(f"   Model: {audio_cfg.get('model', 'N/A')}")
            print(f"   Sample Rate: {audio_cfg.get('sample_rate', 'N/A')}")
            print(f"   Format: {audio_cfg.get('output_format', 'N/A')}")

        # Data payload info
        data_keys = metadata.get('data_keys', [])
        print(f"\n📊 Data Available:")
        print(f"   Keys: {', '.join(data_keys)}")

        # Statistics
        if 'text' in data:
            print(f"   Text Length: {len(data.get('text', ''))} chars")
        if 'chunks' in data:
            print(f"   Chunks: {len(data.get('chunks', []))}")
        if 'processed_chunks' in data:
            print(f"   Processed Chunks: {len(data.get('processed_chunks', []))}")

        # Hash info
        hash_info = metadata.get('hash_info', {})
        if hash_info:
            print(f"\n🔑 Hashes:")
            print(f"   Input Hash: {hash_info.get('input_hash', 'N/A')}")
            print(f"   Config Hash: {hash_info.get('config_hash', 'N/A')}")

        # File size
        file_size = checkpoint_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"\n💾 File Size: {size_mb:.2f} MB")

        print("=" * 80 + "\n")
        return True

    def browse_checkpoints_menu(self) -> Optional[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
        """
        Interactive menu to browse and select a checkpoint.

        Returns:
            Tuple of (checkpoint_path, metadata, data) if selected, None otherwise
        """
        checkpoints = self.list_all_checkpoints()

        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return None

        while True:
            print("\n" + "=" * 80)
            print(f"{'Available Checkpoints':^80}")
            print("=" * 80)

            # Group by input file
            grouped: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
            for path, meta in checkpoints:
                stem = meta.get('input_stem', 'Unknown')
                if stem not in grouped:
                    grouped[stem] = []
                grouped[stem].append((path, meta))

            # Display grouped checkpoints
            idx = 1
            checkpoint_map = {}

            for input_stem, ckpt_list in grouped.items():
                print(f"\n📄 {input_stem}")
                for path, meta in ckpt_list:
                    timestamp = meta.get('timestamp', '')
                    # Parse timestamp for display
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        time_str = timestamp

                    stage = meta.get('stage', '?')
                    config = meta.get('config', {})
                    model = config.get('text_model', 'N/A')

                    print(f"   {idx:2d}. [{stage:10s}] {time_str} | Model: {model}")
                    checkpoint_map[idx] = (path, meta)
                    idx += 1

            print("\n" + "-" * 80)
            print("Options:")
            print("  Enter number to view details")
            print("  'l' - Load selected checkpoint")
            print("  'b' - Back to previous menu")
            print("-" * 80)

            choice = input("Select checkpoint (number) or command: ").strip().lower()

            if choice == 'b':
                return None

            if choice == 'l':
                num_input = input("Enter checkpoint number to load: ").strip()
                try:
                    num = int(num_input)
                    if num in checkpoint_map:
                        path, meta = checkpoint_map[num]
                        # Load full checkpoint data
                        result = self.checkpoint_manager.load(path)
                        if result:
                            metadata, data = result
                            ConsoleOutput.success(f"Loaded checkpoint: {path.name}")
                            return (path, metadata, data)
                        else:
                            ConsoleOutput.error("Failed to load checkpoint data")
                except ValueError:
                    ConsoleOutput.error("Invalid number")
                continue

            try:
                num = int(choice)
                if num in checkpoint_map:
                    path, meta = checkpoint_map[num]
                    self.display_checkpoint_info(path)
                    input("\nPress Enter to continue...")
                else:
                    ConsoleOutput.error("Invalid checkpoint number")
            except ValueError:
                ConsoleOutput.error("Invalid input")

    def checkpoint_management_menu(self):
        """Main checkpoint management menu."""
        while True:
            print("\n" + "=" * 80)
            print(f"{'Checkpoint Management':^80}")
            print("=" * 80)

            # Get checkpoint stats
            checkpoints = self.list_all_checkpoints()
            total_size = sum(path.stat().st_size for path, _ in checkpoints)
            size_mb = total_size / (1024 * 1024)

            print(f"\n📊 Statistics:")
            print(f"   Total Checkpoints: {len(checkpoints)}")
            print(f"   Total Size: {size_mb:.2f} MB")
            print(f"   Location: {self.checkpoint_dir}")

            print("\n" + "-" * 80)
            print("Options:")
            print("  1. Browse Checkpoints")
            print("  2. View Checkpoint Details")
            print("  3. Delete Checkpoint")
            print("  4. Delete All Checkpoints for File")
            print("  5. Delete All Checkpoints")
            print("  6. Change Checkpoint Directory")
            print("  7. Cleanup Old Checkpoints")
            print("  8. Cleanup Corrupted Checkpoints")
            print("  9. Generate Checkpoint Report")
            print("  B. Back to Main Menu")
            print("-" * 80)

            choice = input("Select option: ").strip().upper()

            if choice == 'B':
                break
            elif choice == '1':
                self.browse_checkpoints_menu()
            elif choice == '2':
                self._view_checkpoint_details()
            elif choice == '3':
                self._delete_single_checkpoint()
            elif choice == '4':
                self._delete_checkpoints_for_file()
            elif choice == '5':
                self._delete_all_checkpoints()
            elif choice == '6':
                self._change_checkpoint_directory()
            elif choice == '7':
                self._cleanup_old_checkpoints()
            elif choice == '8':
                self._cleanup_corrupted_checkpoints()
            elif choice == '9':
                self._generate_checkpoint_report()
            else:
                ConsoleOutput.error("Invalid option")

    def _view_checkpoint_details(self):
        """View details of a specific checkpoint."""
        checkpoints = self.list_all_checkpoints()
        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return

        print("\n" + "=" * 80)
        print("Select Checkpoint to View")
        print("=" * 80)

        for idx, (path, meta) in enumerate(checkpoints, 1):
            stem = meta.get('input_stem', 'Unknown')
            stage = meta.get('stage', '?')
            print(f"  {idx}. {stem} - Stage: {stage} - {path.name}")

        choice = input("\nEnter number (or 'b' to go back): ").strip()
        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                path, _ = checkpoints[idx]
                self.display_checkpoint_info(path)
                input("\nPress Enter to continue...")
            else:
                ConsoleOutput.error("Invalid number")
        except ValueError:
            ConsoleOutput.error("Invalid input")

    def _delete_single_checkpoint(self):
        """Delete a single checkpoint file."""
        checkpoints = self.list_all_checkpoints()
        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return

        print("\n" + "=" * 80)
        print("Select Checkpoint to Delete")
        print("=" * 80)

        for idx, (path, meta) in enumerate(checkpoints, 1):
            stem = meta.get('input_stem', 'Unknown')
            stage = meta.get('stage', '?')
            size = path.stat().st_size / 1024  # KB
            print(f"  {idx}. {stem} - {stage} - {path.name} ({size:.1f} KB)")

        choice = input("\nEnter number to delete (or 'b' to cancel): ").strip()
        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                path, meta = checkpoints[idx]
                confirm = input(f"⚠️  Delete {path.name}? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    path.unlink()
                    ConsoleOutput.success(f"Deleted checkpoint: {path.name}")
                else:
                    ConsoleOutput.info("Deletion cancelled")
            else:
                ConsoleOutput.error("Invalid number")
        except ValueError:
            ConsoleOutput.error("Invalid input")
        except Exception as e:
            ConsoleOutput.error(f"Failed to delete checkpoint: {e}")

        input("\nPress Enter to continue...")

    def _delete_checkpoints_for_file(self):
        """Delete all checkpoints for a specific input file."""
        checkpoints = self.list_all_checkpoints()
        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return

        # Group by input file
        grouped: Dict[str, List[Path]] = {}
        for path, meta in checkpoints:
            stem = meta.get('input_stem', 'Unknown')
            if stem not in grouped:
                grouped[stem] = []
            grouped[stem].append(path)

        print("\n" + "=" * 80)
        print("Select Input File to Delete All Checkpoints")
        print("=" * 80)

        files = list(grouped.keys())
        for idx, stem in enumerate(files, 1):
            count = len(grouped[stem])
            total_size = sum(p.stat().st_size for p in grouped[stem]) / (1024 * 1024)  # MB
            print(f"  {idx}. {stem} - {count} checkpoint(s) ({total_size:.2f} MB)")

        choice = input("\nEnter number (or 'b' to cancel): ").strip()
        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                stem = files[idx]
                paths = grouped[stem]
                confirm = input(f"⚠️  Delete all {len(paths)} checkpoint(s) for '{stem}'? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    for path in paths:
                        path.unlink()
                    ConsoleOutput.success(f"Deleted {len(paths)} checkpoint(s)")

                    # Clean up empty directories
                    for path in paths:
                        parent = path.parent
                        if parent.exists() and not any(parent.iterdir()):
                            parent.rmdir()
                else:
                    ConsoleOutput.info("Deletion cancelled")
            else:
                ConsoleOutput.error("Invalid number")
        except ValueError:
            ConsoleOutput.error("Invalid input")
        except Exception as e:
            ConsoleOutput.error(f"Failed to delete checkpoints: {e}")

        input("\nPress Enter to continue...")

    def _delete_all_checkpoints(self):
        """Delete all checkpoints."""
        checkpoints = self.list_all_checkpoints()
        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return

        total_size = sum(p.stat().st_size for p, _ in checkpoints) / (1024 * 1024)

        print("\n" + "=" * 80)
        print(f"⚠️  WARNING: Delete All Checkpoints")
        print("=" * 80)
        print(f"\nThis will delete {len(checkpoints)} checkpoint(s) ({total_size:.2f} MB)")
        print(f"Location: {self.checkpoint_dir}")

        confirm = input("\nType 'DELETE ALL' to confirm: ").strip()
        if confirm == 'DELETE ALL':
            self.checkpoint_manager.delete_all_checkpoints()
            ConsoleOutput.success("All checkpoints deleted")
        else:
            ConsoleOutput.info("Deletion cancelled")

        input("\nPress Enter to continue...")

    def _change_checkpoint_directory(self):
        """Change the checkpoint directory location."""
        print("\n" + "=" * 80)
        print("Change Checkpoint Directory")
        print("=" * 80)
        print(f"\nCurrent directory: {self.checkpoint_dir}")
        print("\nNote: This only changes the directory for this session.")
        print("To permanently change it, update the configuration.")

        new_dir = input("\nEnter new directory path (or 'b' to cancel): ").strip()
        if new_dir.lower() == 'b':
            return

        new_path = Path(new_dir).expanduser().resolve()
        if new_path.exists() and new_path.is_dir():
            self.checkpoint_dir = new_path
            self.checkpoint_manager = CheckpointManager(base_checkpoint_dir=new_path)
            ConsoleOutput.success(f"Checkpoint directory changed to: {new_path}")
        else:
            create = input(f"Directory doesn't exist. Create it? (y/n): ").strip().lower()
            if create == 'y':
                new_path.mkdir(parents=True, exist_ok=True)
                self.checkpoint_dir = new_path
                self.checkpoint_manager = CheckpointManager(base_checkpoint_dir=new_path)
                ConsoleOutput.success(f"Created and set checkpoint directory: {new_path}")
            else:
                ConsoleOutput.info("Directory change cancelled")

        input("\nPress Enter to continue...")

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only recent ones."""
        print("\n" + "=" * 80)
        print("Cleanup Old Checkpoints")
        print("=" * 80)

        keep_num = input("\nHow many recent checkpoints to keep per stage? (default: 3): ").strip()
        try:
            keep_num = int(keep_num) if keep_num else 3
        except ValueError:
            keep_num = 3

        # Group by input file
        checkpoints = self.list_all_checkpoints()
        grouped: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
        for path, meta in checkpoints:
            stem = meta.get('input_stem', 'Unknown')
            if stem not in grouped:
                grouped[stem] = []
            grouped[stem].append((path, meta))

        total_deleted = 0
        for stem, ckpt_list in grouped.items():
            # Group by stage
            by_stage: Dict[str, List[Tuple[Path, str]]] = {}
            for path, meta in ckpt_list:
                stage = meta.get('stage', 'unknown')
                timestamp = meta.get('timestamp', '')
                if stage not in by_stage:
                    by_stage[stage] = []
                by_stage[stage].append((path, timestamp))

            # Sort and delete old ones
            for stage, stage_ckpts in by_stage.items():
                # Sort by timestamp (newest first)
                stage_ckpts.sort(key=lambda x: x[1], reverse=True)

                # Delete old ones
                for path, _ in stage_ckpts[keep_num:]:
                    try:
                        path.unlink()
                        total_deleted += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {path.name}: {e}")

        ConsoleOutput.success(f"Cleaned up {total_deleted} old checkpoint(s)")
        input("\nPress Enter to continue...")

    def _cleanup_corrupted_checkpoints(self):
        """Clean up corrupted checkpoint files."""
        print("\n" + "=" * 80)
        print("Cleanup Corrupted Checkpoints")
        print("=" * 80)
        print("\nScanning for corrupted or unreadable checkpoint files...")

        deleted = self.checkpoint_manager.cleanup_corrupted_checkpoints()

        if deleted > 0:
            ConsoleOutput.success(f"Cleaned up {deleted} corrupted checkpoint(s)")
        else:
            ConsoleOutput.info("No corrupted checkpoints found!")

        input("\nPress Enter to continue...")

    def _generate_checkpoint_report(self):
        """Generate a detailed report for a checkpoint."""
        checkpoints = self.list_all_checkpoints()
        if not checkpoints:
            ConsoleOutput.warning("No checkpoints found.")
            input("\nPress Enter to continue...")
            return

        print("\n" + "=" * 80)
        print("Select Checkpoint for Detailed Report")
        print("=" * 80)

        for idx, (path, meta) in enumerate(checkpoints, 1):
            stem = meta.get('input_stem', 'Unknown')
            stage = meta.get('stage', '?')
            timestamp = meta.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = timestamp
            print(f"  {idx}. {stem} - {stage} - {time_str}")

        choice = input("\nEnter number (or 'b' to go back): ").strip()
        if choice.lower() == 'b':
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                path, _ = checkpoints[idx]
                report = self.checkpoint_manager.generate_checkpoint_report(path)
                print("\n" + report)

                # Ask if user wants to save report
                save = input("\nSave report to file? (y/n): ").strip().lower()
                if save == 'y':
                    report_path = path.with_suffix('.txt')
                    report_path.write_text(report)
                    ConsoleOutput.success(f"Report saved to: {report_path}")

                input("\nPress Enter to continue...")
            else:
                ConsoleOutput.error("Invalid number")
        except ValueError:
            ConsoleOutput.error("Invalid input")
        except Exception as e:
            ConsoleOutput.error(f"Failed to generate report: {e}")
            input("\nPress Enter to continue...")
