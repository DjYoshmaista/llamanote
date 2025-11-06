llamanote/
├── __init__.py              # Package initialization
├── main.py                  # Main CLI entry point
├── config.py                # Configuration settings
├── core/
│   ├── __init__.py
│   ├── pdf_processor.py     # PDF extraction and metadata
│   ├── text_processor.py    # Chunking and text processing
│   ├── model_manager.py     # Model loading and optimization
│   └── generator.py         # LLM generation with filtering
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Centralized logging
│   ├── validators.py        # Input validation
│   └── formatters.py        # Output formatting (markdown, etc.)
└── filters/
    ├── __init__.py
    └── thinking_filter.py   # Filter thinking tokens
