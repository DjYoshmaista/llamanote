# llamanote/utils/file_browser.py
"""
Simple file browser utility for interactive file selection.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple

from .logger import ConsoleOutput


class FileBrowser:
    """Simple interactive file browser for the CLI."""

    @staticmethod
    def browse_directory(
        start_dir: Path,
        file_extensions: Optional[List[str]] = None,
        title: str = "Select File"
    ) -> Optional[Path]:
        """
        Browse directory and select a file.

        Args:
            start_dir: Directory to start browsing from
            file_extensions: List of allowed extensions (e.g., ['.pdf', '.txt']), None for all
            title: Title to display

        Returns:
            Selected file path or None if cancelled
        """
        current_dir = Path(start_dir).expanduser().resolve()

        if not current_dir.exists():
            ConsoleOutput.error(f"Directory does not exist: {current_dir}")
            return None

        if not current_dir.is_dir():
            current_dir = current_dir.parent

        while True:
            # List directory contents
            try:
                items = sorted(current_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except PermissionError:
                ConsoleOutput.error(f"Permission denied: {current_dir}")
                return None

            # Filter files by extension and exclude hidden files
            dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            files = [item for item in items if item.is_file() and not item.name.startswith('.')]

            if file_extensions:
                files = [f for f in files if f.suffix.lower() in file_extensions]

            print("\n" + "=" * 80)
            print(f"{title:^80}")
            print("=" * 80)
            print(f"\n📁 Current Directory: {current_dir}")
            print("\nDirectories:")

            idx = 1
            item_map = {}

            # Parent directory option
            if current_dir.parent != current_dir:
                print(f"  0. 📁 ..")
                item_map[0] = current_dir.parent

            # List directories
            for d in dirs[:20]:  # Limit to 20 dirs
                print(f"  {idx:2d}. 📁 {d.name}")
                item_map[idx] = d
                idx += 1

            # List files
            if files:
                print("\nFiles:")
                for f in files[:30]:  # Limit to 30 files
                    size = f.stat().st_size / 1024  # KB
                    print(f"  {idx:2d}. 📄 {f.name} ({size:.1f} KB)")
                    item_map[idx] = f
                    idx += 1

            if not dirs and not files:
                print("\n  (Empty directory)")

            print("\n" + "-" * 80)
            print("Options:")
            print("  Enter number to select")
            print("  'p' - Enter custom path")
            print("  'b' - Back/Cancel")
            print("-" * 80)

            choice = input("Select: ").strip().lower()

            if choice == 'b':
                return None
            elif choice == 'p':
                custom_path = input("Enter file path: ").strip()
                path = Path(custom_path).expanduser().resolve()
                if path.exists() and path.is_file():
                    if file_extensions is None or path.suffix.lower() in file_extensions:
                        return path
                    else:
                        ConsoleOutput.error(f"Invalid file type. Allowed: {', '.join(file_extensions)}")
                else:
                    ConsoleOutput.error("File does not exist")
                input("\nPress Enter to continue...")
                continue

            try:
                num = int(choice)
                if num in item_map:
                    selected = item_map[num]
                    if selected.is_dir():
                        current_dir = selected
                    else:
                        # File selected
                        return selected
                else:
                    ConsoleOutput.error("Invalid number")
            except ValueError:
                ConsoleOutput.error("Invalid input")

    @staticmethod
    def select_file_type_and_browse() -> Optional[Tuple[Path, str]]:
        """
        Show file type selection menu and browse accordingly.

        Returns:
            Tuple of (selected_path, file_type) or None if cancelled
        """
        while True:
            print("\n" + "=" * 80)
            print(f"{'Select Input Type':^80}")
            print("=" * 80)
            print("\n1. PDF Document")
            print("2. Text File (.txt)")
            print("3. Markdown File (.md)")
            print("4. Transcript/Formatted File (from output/)")
            print("5. Checkpoint File (.ckpt)")
            print("6. Enter Custom Path")
            print("B. Back")
            print("-" * 80)

            choice = input("Select option: ").strip().upper()

            if choice == 'B':
                return None
            elif choice == '1':
                # PDF - start from home
                path = FileBrowser.browse_directory(
                    Path.home(),
                    file_extensions=['.pdf'],
                    title="Select PDF File"
                )
                if path:
                    return (path, 'pdf')
            elif choice == '2':
                # Text file - start from output
                output_dir = Path.cwd() / "output"
                if not output_dir.exists():
                    output_dir = Path.home()
                path = FileBrowser.browse_directory(
                    output_dir,
                    file_extensions=['.txt'],
                    title="Select Text File"
                )
                if path:
                    return (path, 'text')
            elif choice == '3':
                # Markdown - start from output
                output_dir = Path.cwd() / "output"
                if not output_dir.exists():
                    output_dir = Path.home()
                path = FileBrowser.browse_directory(
                    output_dir,
                    file_extensions=['.md'],
                    title="Select Markdown File"
                )
                if path:
                    return (path, 'markdown')
            elif choice == '4':
                # Transcript - start from output
                output_dir = Path.cwd() / "output"
                if not output_dir.exists():
                    output_dir = Path.home()
                path = FileBrowser.browse_directory(
                    output_dir,
                    file_extensions=['.md', '.txt'],
                    title="Select Transcript File"
                )
                if path:
                    return (path, 'transcript')
            elif choice == '5':
                # Checkpoint - start from checkpoints folder
                checkpoint_dir = Path.cwd() / "checkpoints"
                if not checkpoint_dir.exists():
                    checkpoint_dir = Path.home()
                path = FileBrowser.browse_directory(
                    checkpoint_dir,
                    file_extensions=['.ckpt'],
                    title="Select Checkpoint File"
                )
                if path:
                    return (path, 'checkpoint')
            elif choice == '6':
                # Custom path
                custom_path = input("\nEnter file path: ").strip()
                path = Path(custom_path).expanduser().resolve()
                if path.exists() and path.is_file():
                    # Determine type by extension
                    ext = path.suffix.lower()
                    if ext == '.pdf':
                        file_type = 'pdf'
                    elif ext == '.ckpt':
                        file_type = 'checkpoint'
                    elif ext in ['.txt', '.md']:
                        # Ask if it's a transcript
                        is_transcript = input("Is this a transcript/formatted file? (y/n): ").strip().lower()
                        file_type = 'transcript' if is_transcript == 'y' else ('markdown' if ext == '.md' else 'text')
                    else:
                        file_type = 'unknown'
                    return (path, file_type)
                else:
                    ConsoleOutput.error("File does not exist")
                    input("\nPress Enter to continue...")
            else:
                ConsoleOutput.error("Invalid option")


# Import for type hinting
from typing import Tuple
