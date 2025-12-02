"""
Checkpoint Registry - CSV-based tracking system for fast checkpoint discovery.

The registry provides a centralized index of all checkpoints, enabling:
- Fast filtering and searching without loading checkpoint files
- Sortable views by any column (progress, model, stage, etc.)
- Efficient compatibility checking
- Stale entry detection and cleanup
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


class CheckpointRegistry:
    """
    Manages a CSV registry of checkpoints for fast discovery and filtering.

    The registry is stored as a hidden .checkpoints.csv file in the checkpoint directory.
    Each row represents one checkpoint with key metadata for quick access.

    CSV Schema:
        filename, input_file, input_stem, file_extension, stage, chunk_index,
        total_chunks, completion_pct, text_model, audio_model,
        hyperparameter_preset, created_timestamp, size_mb,
        compressed_size_mb, hash, status
    """

    # CSV column definitions
    COLUMNS = [
        "filename",
        "input_file",
        "input_stem",
        "file_extension",
        "stage",
        "chunk_index",
        "total_chunks",
        "completion_pct",
        "text_model",
        "audio_model",
        "hyperparameter_preset",
        "created_timestamp",
        "size_mb",
        "compressed_size_mb",
        "hash",
        "status"
    ]

    def __init__(self, registry_path: Optional[Path] = None, checkpoint_dir: Optional[Path] = None):
        """
        Initialize the checkpoint registry.

        Args:
            registry_path: Path to the registry CSV file. If None, uses default location.
            checkpoint_dir: Checkpoint directory. Used to derive default registry path.
        """
        if registry_path is None:
            if checkpoint_dir is None:
                checkpoint_dir = Path("checkpoints")
            self.registry_path = checkpoint_dir / ".checkpoints.csv"
        else:
            self.registry_path = Path(registry_path)

        self.logger = logger
        self._lock = threading.RLock()  # Thread-safe operations (reentrant lock to avoid deadlock)

        # Ensure registry file exists
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        """Create registry file with header if it doesn't exist."""
        if not self.registry_path.exists():
            try:
                self.registry_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.registry_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                    writer.writeheader()
                self.logger.debug(f"Created new registry: {self.registry_path}")
            except Exception as e:
                self.logger.error(f"Failed to create registry: {e}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Load all entries from the registry.

        Returns:
            List of dictionaries, one per checkpoint entry
        """
        with self._lock:
            entries = []

            if not self.registry_path.exists():
                return entries

            try:
                with open(self.registry_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert numeric fields
                        if row.get('chunk_index'):
                            row['chunk_index'] = int(row['chunk_index']) if row['chunk_index'] != 'None' else None
                        if row.get('total_chunks'):
                            row['total_chunks'] = int(row['total_chunks']) if row['total_chunks'] != 'None' else None
                        if row.get('completion_pct'):
                            row['completion_pct'] = int(row['completion_pct']) if row['completion_pct'] != 'None' else None
                        if row.get('size_mb'):
                            row['size_mb'] = float(row['size_mb']) if row['size_mb'] else 0.0
                        if row.get('compressed_size_mb'):
                            row['compressed_size_mb'] = float(row['compressed_size_mb']) if row['compressed_size_mb'] else 0.0

                        entries.append(row)

                self.logger.debug(f"Loaded {len(entries)} entries from registry")
                return entries

            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                return []

    def save(self, entries: List[Dict[str, Any]]) -> bool:
        """
        Save all entries to the registry (overwrites existing file).

        Args:
            entries: List of checkpoint entry dictionaries

        Returns:
            True if save succeeded, False otherwise
        """
        with self._lock:
            try:
                # Atomic write: write to temp file first
                temp_path = self.registry_path.with_suffix('.tmp')

                with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                    writer.writeheader()
                    writer.writerows(entries)

                # Atomic rename
                temp_path.replace(self.registry_path)
                self.logger.debug(f"Saved {len(entries)} entries to registry")
                return True

            except Exception as e:
                self.logger.error(f"Failed to save registry: {e}")
                # Clean up temp file
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                return False

    def add_entry(self, checkpoint_info: Dict[str, Any]) -> bool:
        """
        Add a new checkpoint entry to the registry.

        Args:
            checkpoint_info: Dictionary with checkpoint metadata

        Returns:
            True if entry added successfully
        """
        with self._lock:
            try:
                # Load existing entries
                entries = self.load()

                # Check if entry already exists (by filename)
                filename = checkpoint_info.get('filename')
                if filename:
                    # Remove any existing entry with same filename
                    entries = [e for e in entries if e.get('filename') != filename]

                # Add new entry
                entry = self._normalize_entry(checkpoint_info)
                entries.append(entry)

                # Save back
                return self.save(entries)

            except Exception as e:
                self.logger.error(f"Failed to add entry: {e}")
                return False

    def remove_entry(self, checkpoint_filename: str) -> bool:
        """
        Remove a checkpoint entry from the registry.

        Args:
            checkpoint_filename: Name of the checkpoint file

        Returns:
            True if entry removed successfully
        """
        with self._lock:
            try:
                entries = self.load()
                original_count = len(entries)

                # Filter out the entry
                entries = [e for e in entries if e.get('filename') != checkpoint_filename]

                if len(entries) < original_count:
                    self.logger.debug(f"Removed entry: {checkpoint_filename}")
                    return self.save(entries)
                else:
                    self.logger.warning(f"Entry not found: {checkpoint_filename}")
                    return False

            except Exception as e:
                self.logger.error(f"Failed to remove entry: {e}")
                return False

    def update_entry(self, checkpoint_filename: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing checkpoint entry.

        Args:
            checkpoint_filename: Name of the checkpoint file
            updates: Dictionary of fields to update

        Returns:
            True if entry updated successfully
        """
        with self._lock:
            try:
                entries = self.load()
                updated = False

                for entry in entries:
                    if entry.get('filename') == checkpoint_filename:
                        entry.update(updates)
                        updated = True
                        break

                if updated:
                    return self.save(entries)
                else:
                    self.logger.warning(f"Entry not found for update: {checkpoint_filename}")
                    return False

            except Exception as e:
                self.logger.error(f"Failed to update entry: {e}")
                return False

    def find_by_input(self, input_stem: str) -> List[Dict[str, Any]]:
        """
        Find all checkpoints for a specific input file.

        Args:
            input_stem: Input file stem (filename without extension)

        Returns:
            List of matching checkpoint entries
        """
        entries = self.load()
        return [e for e in entries if e.get('input_stem') == input_stem]

    def find_by_stage(self, stage: str) -> List[Dict[str, Any]]:
        """
        Find all checkpoints for a specific stage.

        Args:
            stage: Stage name (e.g., 'process', 'audio')

        Returns:
            List of matching checkpoint entries
        """
        entries = self.load()
        return [e for e in entries if e.get('stage') == stage]

    def find_by_model(self, model_abbrev: str) -> List[Dict[str, Any]]:
        """
        Find all checkpoints using a specific model.

        Args:
            model_abbrev: Model abbreviation (text or audio)

        Returns:
            List of matching checkpoint entries
        """
        entries = self.load()
        return [
            e for e in entries
            if e.get('text_model') == model_abbrev or e.get('audio_model') == model_abbrev
        ]

    def get_latest_for_input(self, input_stem: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for an input file.

        Args:
            input_stem: Input file stem

        Returns:
            Latest checkpoint entry or None
        """
        matching = self.find_by_input(input_stem)
        if not matching:
            return None

        # Sort by timestamp (newest first)
        matching.sort(key=lambda x: x.get('created_timestamp', ''), reverse=True)
        return matching[0]

    def verify_checkpoints_exist(self, checkpoint_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Verify that all registry entries have corresponding checkpoint files.

        Args:
            checkpoint_dir: Directory containing checkpoint subdirectories

        Returns:
            Tuple of (existing_filenames, missing_filenames)
        """
        entries = self.load()
        existing = []
        missing = []

        for entry in entries:
            filename = entry.get('filename')
            if not filename:
                continue

            # Search for checkpoint file in subdirectories
            found = False
            for ckpt_file in checkpoint_dir.rglob(filename):
                if ckpt_file.name == filename:
                    existing.append(filename)
                    found = True
                    break

            if not found:
                missing.append(filename)

        return existing, missing

    def cleanup_missing_entries(self, checkpoint_dir: Path) -> int:
        """
        Remove registry entries for checkpoints that no longer exist.

        Args:
            checkpoint_dir: Directory containing checkpoint subdirectories

        Returns:
            Number of entries removed
        """
        with self._lock:
            _, missing = self.verify_checkpoints_exist(checkpoint_dir)

            if not missing:
                return 0

            # Remove missing entries
            entries = self.load()
            original_count = len(entries)
            entries = [e for e in entries if e.get('filename') not in missing]

            self.save(entries)
            removed_count = original_count - len(entries)

            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} stale registry entries")

            return removed_count

    def _normalize_entry(self, checkpoint_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Normalize checkpoint info to registry entry format.

        Ensures all required columns are present and values are strings.

        Args:
            checkpoint_info: Raw checkpoint information

        Returns:
            Normalized entry dictionary
        """
        entry = {}

        for col in self.COLUMNS:
            value = checkpoint_info.get(col)

            # Convert to string, handling None and empty values
            if value is None or value == '':
                entry[col] = 'None'
            else:
                entry[col] = str(value)

        return entry

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with statistics (total, by_stage, by_status, etc.)
        """
        entries = self.load()

        stats = {
            "total_entries": len(entries),
            "by_stage": {},
            "by_status": {},
            "by_input": {},
            "total_size_mb": 0.0
        }

        for entry in entries:
            # Count by stage
            stage = entry.get('stage', 'unknown')
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1

            # Count by status
            status = entry.get('status', 'active')
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by input
            input_stem = entry.get('input_stem', 'unknown')
            stats["by_input"][input_stem] = stats["by_input"].get(input_stem, 0) + 1

            # Sum sizes
            try:
                size_mb = float(entry.get('compressed_size_mb', 0))
                stats["total_size_mb"] += size_mb
            except (ValueError, TypeError):
                pass

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)

        return stats

    def export_to_json(self, output_path: Path) -> bool:
        """
        Export registry to JSON format.

        Args:
            output_path: Path to write JSON file

        Returns:
            True if export successful
        """
        try:
            import json

            # Load entries
            entries = self.load()

            # Add metadata
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(entries),
                "checkpoints": entries
            }

            # Write JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"Exported {len(entries)} entries to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            return False

    def export_to_html(self, output_path: Path) -> bool:
        """
        Export registry to HTML format with sortable table.

        Args:
            output_path: Path to write HTML file

        Returns:
            True if export successful
        """
        try:
            entries = self.load()
            stats = self.get_statistics()

            # Build HTML
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkpoint Registry Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}
        th:hover {{
            background-color: #0056b3;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .filename {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .progress-bar {{
            width: 100px;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #28a745;
            transition: width 0.3s;
        }}
        .status-active {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-legacy {{
            color: #ffc107;
            font-weight: bold;
        }}
    </style>
    <script>
        function sortTable(columnIndex) {{
            const table = document.getElementById('checkpointTable');
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            const isNumeric = columnIndex === 5 || columnIndex === 6 || columnIndex === 7;

            rows.sort((a, b) => {{
                const aVal = a.cells[columnIndex].textContent;
                const bVal = b.cells[columnIndex].textContent;

                if (isNumeric) {{
                    return parseFloat(aVal || 0) - parseFloat(bVal || 0);
                }}
                return aVal.localeCompare(bVal);
            }});

            const tbody = table.querySelector('tbody');
            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>Checkpoint Registry Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Total Checkpoints</div>
                <div class="stat-value">{stats['total_entries']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Size</div>
                <div class="stat-value">{stats['total_size_mb']} MB</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Active</div>
                <div class="stat-value">{stats['by_status'].get('active', 0)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Legacy</div>
                <div class="stat-value">{stats['by_status'].get('legacy', 0)}</div>
            </div>
        </div>

        <table id="checkpointTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Filename ▾</th>
                    <th onclick="sortTable(1)">Input File ▾</th>
                    <th onclick="sortTable(2)">Stage ▾</th>
                    <th onclick="sortTable(3)">Progress ▾</th>
                    <th onclick="sortTable(4)">Model ▾</th>
                    <th onclick="sortTable(5)">Chunk ▾</th>
                    <th onclick="sortTable(6)">Size (MB) ▾</th>
                    <th onclick="sortTable(7)">Created ▾</th>
                    <th onclick="sortTable(8)">Status ▾</th>
                </tr>
            </thead>
            <tbody>
"""

            # Add table rows
            for entry in entries:
                completion = entry.get('completion_pct', 'N/A')
                chunk_info = f"{entry.get('chunk_index', 'N/A')}/{entry.get('total_chunks', 'N/A')}"
                size_mb = float(entry.get('compressed_size_mb', 0))
                status = entry.get('status', 'active')
                status_class = f"status-{status}"

                # Progress bar
                progress_html = ""
                if completion != 'N/A' and completion != 'None':
                    try:
                        pct = int(completion)
                        progress_html = f"""
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {pct}%"></div>
                        </div>
                        {pct}%
                        """
                    except:
                        progress_html = "N/A"
                else:
                    progress_html = "N/A"

                html += f"""
                <tr>
                    <td class="filename">{entry.get('filename', '')[:50]}...</td>
                    <td>{entry.get('input_stem', '')}</td>
                    <td>{entry.get('stage', '')}</td>
                    <td>{progress_html}</td>
                    <td>{entry.get('text_model', 'unknown')}</td>
                    <td>{chunk_info}</td>
                    <td>{size_mb:.2f}</td>
                    <td>{entry.get('created_timestamp', '')[:19]}</td>
                    <td class="{status_class}">{status}</td>
                </tr>
"""

            html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

            # Write HTML
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)

            self.logger.info(f"Exported HTML report to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export to HTML: {e}")
            return False
