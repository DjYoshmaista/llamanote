#!/usr/bin/env python3
"""
LlamaNote Enhanced - Main Entry Point

This script parses command-line arguments and either launches the
interactive menu system or executes a processing task directly via the CLI.
"""

import sys
import logging
from src.cli import parse_arguments, handle_utility_commands, run_cli_processing
from src.menu import MenuSystem
from src.utils.logger import setup_logging, ConsoleOutput, get_logger_conf

# Add folder to path for the current session
sys.path.append('.')

# Initialize a basic logger for the main script
# setup_logging() is called in __init__.py, but we grab the logger here.
logger = get_logger_conf("llamanote_main")

def main():
    """Main execution function: parses args, runs utilities, CLI, or Menu."""
    try:
        parser = parse_arguments()
        args = parser.parse_args()

        # --- 1. Set Verbosity ---
        log_level = logging.DEBUG if args.verbose else logging.INFO
        # We need to update the level for the logger and its handlers
        logging.getLogger("llamanote").setLevel(log_level)
        # Find the console handler and update its level
        for handler in logging.getLogger("llamanote").handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(log_level)
                break
        
        logger.debug("Debug logging enabled.")

        # --- 2. Handle Utility Commands (List models, etc.) ---
        # These commands run *instead* of the main pipeline or menu.
        if handle_utility_commands(args, parser):
            return 0 # Utility command was run, exit successfully

        # --- 3. Decide: CLI Mode or Menu Mode ---
        if args.input:
            # If input files ARE provided, run CLI processing
            logger.info("Input files provided. Running in command-line mode...")
            exit_code = run_cli_processing(args, parser)
            return exit_code
        else:
            # If NO input files and NO utility commands were run, start the menu
            logger.info("No input files specified. Starting interactive menu...")
            menu = MenuSystem()
            menu.run() # This blocks until the menu exits
            return 0

    except Exception as e:
        ConsoleOutput.error(f"An unexpected critical error occurred: {e}")
        logger.critical("LlamaNote failed to run.", exc_info=True)
        # Optionally save error context if logger is configured
        if hasattr(logger, 'error'):
             logger.error(f"Critical failure in main: {e}", exc_info=True, save_context=True)
        return 1
    except KeyboardInterrupt:
        ConsoleOutput.warning("\n\nOperation cancelled by user.")
        return 130 # Standard exit code for Ctrl+C

if __name__ == "__main__":
    sys.exit(main())
