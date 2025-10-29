llamanote/
├── __init__.py              # Package initialization
├── main.py                  # Main CLI/Menu entry point
├── src/
│   ├── __init__.py
│   ├── cli.py               # Command Line Interface logic
│   ├── menu.py              # Interactive Menu System logic
│   ├── config/              # Configuration files (settings, profiles, keys, manager)
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── profiles.py
│   │   ├── presets.py
│   │   ├── cloud_keys.py
│   │   └── manager.py
│   ├── core/                # Core pipeline, stages, types, errors
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── errors.py
│   │   ├── pipeline.py
│   │   └── stages.py
│   ├── formatting/          # Output formatters (Markdown, Podcast, etc.)
│   │   ├── __init__.py
│   │   ├── base_formatter.py
│   │   ├── podcast_formatter.py
│   │   └── ... (other formatters)
│   ├── io/                  # Input/Output handling (files, checkpoints, batch)
│   │   ├── __init__.py
│   │   ├── file_handler.py
│   │   ├── batch_manager.py
│   │   └── checkpoints.py
│   ├── models/              # AI model handling (registry, hub, backends)
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── hub.py
│   │   ├── hyperparameters.py
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── local_hf.py
│   │       └── ... (other backends)
│   ├── processing/          # Individual processing steps (PDF, text, audio)
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py
│   │   ├── text_preprocessor.py
│   │   ├── text_chunker.py
│   │   ├── response_filter.py
│   │   └── audio_processor.py
│   └── utils/               # Utility functions (logger, validators, helpers)
│       ├── __init__.py
│       ├── logger.py
│       ├── validators.py
│       ├── decorators.py
│       └── helpers.py
├── presets/                 # JSON preset files (hyperparams, models)
├── output/                  # Default output directory (ignored)
├── logs/                    # Default log directory (ignored)
├── cache/                   # Default cache directory (ignored)
├── README.md
├── requirements.txt         # Project dependencies
└── .gitignore
