# llamanote/models/hub.py
"""
Model Hub Module
Manages browsing, searching, and downloading models from HuggingFace Hub.
Refactored for clarity and error handling.
"""

import os
import json
import re
import requests
import time
from functools import wraps
from pathlib import Path
from typing import List, Dict, Callable, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, GatedRepoError
from tqdm import tqdm # Assuming tqdm is a dependency

from ..utils.logger import get_logger_conf, ConsoleOutput # Use new logger
from ..utils.decorators import log_execution_time
from ..config.settings import DEFAULT_CACHE_DIR # Use central settings
from .registry import get_registry

logger = get_logger_conf(__name__)

# --- Dataclass (aliased in types.py, but defined here) ---
# Note: This is aliased as HFModelInfo in menu_system.py to avoid confusion with ModelEntry
@dataclass
class ModelHubInfo:
    """Standardized information about a HuggingFace model."""
    model_id: str
    author: str
    model_name: str
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    model_size_mb: Optional[float] = None # Estimated size
    description: Optional[str] = None
    last_modified: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        size_str = f" ({self.format_size()})" if self.model_size_mb else ""
        return f"{self.model_id}{size_str}"
    
    def format_size(self) -> str:
        """Format model size for display."""
        if self.model_size_mb is None:
            return "Size unknown"
        
        if self.model_size_mb < 1024:
            return f"{self.model_size_mb:.1f} MB"
        else:
            return f"{self.model_size_mb / 1024:.1f} GB"

    def matches_filter(self, 
                      search_term: Optional[str] = None,
                      min_downloads: int = 0,
                      library: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> bool:
        """Check if model matches filter criteria (Used by ModelHub.search_models)."""
        if search_term:
            search_lower = search_term.lower()
            if not (search_lower in self.model_id.lower() or 
                   (self.description and search_lower in self.description.lower())):
                return False
        
        if self.downloads < min_downloads:
            return False
        
        if library and self.library_name != library:
            return False
        
        if tags:
            # Check if all specified tags are present
            if not all(tag.lower() in [t.lower() for t in self.tags] for tag in tags):
                 return False
        
        return True


# --- Cache Helper (Refactoring Item 7) ---
class CacheManager:
    """Manages simple JSON caching for model info."""
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads the cache from a JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load model hub cache: {e}. Starting fresh.")
                return {}
        return {}

    def save_cache(self, cache_data: Dict[str, Dict[str, Any]]):
        """Saves the cache to a JSON file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved {len(cache_data)} items to model hub cache.")
        except Exception as e:
            logger.warning(f"Failed to save model hub cache: {e}")

# --- Decorator (Refactoring Item 7) ---
def handle_hub_errors(func: Callable) -> Callable:
    """Decorator to catch common HfApi errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RepositoryNotFoundError:
            logger.warning(f"Repository not found for {args[1:]}")
            ConsoleOutput.warning(f"Model repository not found.")
            return None
        except GatedRepoError:
            logger.warning(f"Repository is gated. User must log in or accept terms on Hub.")
            ConsoleOutput.warning(f"Model repository is gated. Please visit the Hub page and accept terms.")
            return None
        except HfHubHTTPError as e:
            logger.error(f"HuggingFace API request failed: {e}", exc_info=True)
            ConsoleOutput.error(f"HuggingFace API error: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
             logger.error(f"Network connection error to HuggingFace Hub: {e}")
             ConsoleOutput.error(f"Network error: Could not connect to HuggingFace Hub.")
             return None
        except Exception as e:
            logger.error(f"An unexpected error occurred in {func.__name__}: {e}", exc_info=True)
            ConsoleOutput.error(f"An unexpected error occurred: {e}")
            return None
    return wrapper

# --- Model Size Estimator (Refactoring Item 7) ---
class ModelSizeEstimator:
    """Estimates model size based on name/tags or file listings."""
    # (Simplified version - full file listing is slow for searching)
    
    # Common size indicators (param count * approx bytes per param)
    # Assuming 2 bytes/param (float16/bfloat16) for base size
    _SIZE_INDICATORS = {
        '270m': 270 * 2,
        '1b': 1 * 1024,
        '1.5b': 1.5 * 1024,
        '3b': 3 * 1024,
        '4b': 4 * 1024,
        '7b': 7 * 1024,
        '8b': 8 * 1024,
        '13b': 13 * 1024,
        '30b': 30 * 1024,
        '34b': 34 * 1024,
        '65b': 65 * 1024,
        '70b': 70 * 1024,
        'q2': 0.3, # Quantization multipliers (approx)
        'q3': 0.4,
        'q4': 0.5,
        'q5': 0.6,
        'q6': 0.7,
        'q8': 0.9,
        'f16': 2.0 / 2.0, # (Base is fp16/bf16)
        'f32': 4.0 / 2.0
    }
    # Sort by length descending to match longer keys first
    _SORTED_SIZE_KEYS = sorted(_SIZE_INDICATORS.keys(), key=len, reverse=True)

    @classmethod
    def estimate_from_name_tags(cls, model_id: str, tags: List[str]) -> Optional[float]:
        """Estimates size in MB based on model_id and tags."""
        name_lower = f"{model_id.lower()} {' '.join(tags).lower()}"
        
        base_size_mb: Optional[float] = None
        quant_multiplier: Optional[float] = None

        for key in cls._SORTED_SIZE_KEYS:
            if re.search(r'[\W_]' + re.escape(key) + r'[\W_]', name_lower) or \
               name_lower.endswith('-' + key) or \
               name_lower.startswith(key + '-'):
                
                size_val = cls._SIZE_INDICATORS[key]
                if 'b' in key: # It's a base parameter size
                    if base_size_mb is None: # Take first one found
                         base_size_mb = size_val
                else: # It's a quantization multiplier
                    if quant_multiplier is None: # Take first one found
                         quant_multiplier = size_val
        
        if base_size_mb is not None:
             # Apply quantization multiplier if found, otherwise assume full (or fp16/bf16)
             multiplier = quant_multiplier or 1.0 
             return base_size_mb * multiplier
             
        return None # Could not determine base size

# --- Model Info Extractor (Refactoring Item 7) ---
class ModelInfoExtractor:
    """Extracts standardized ModelHubInfo from HuggingFace API ModelInfo object."""
    @staticmethod
    def extract(model_info_obj: ModelInfo, api: Optional[HfApi] = None) -> Optional[ModelHubInfo]:
        """Converts HfApi.ModelInfo to our local ModelHubInfo."""
        try:
            model_id = model_info_obj.modelId
            parts = model_id.split('/')
            author = parts[0] if len(parts) > 1 else "unknown"
            model_name = parts[-1]
            
            tags = getattr(model_info_obj, 'tags', []) or []
            
            # Try to get description (requires extra API call if not in list_models result)
            description = None
            if hasattr(model_info_obj, 'cardData') and model_info_obj.cardData:
                 description = str(model_info_obj.cardData.get('description', ''))[:200]
            elif api: # If API client passed, try to fetch full info
                 try:
                      full_info = api.model_info(model_id)
                      if hasattr(full_info, 'cardData') and full_info.cardData:
                           description = str(full_info.cardData.get('description', ''))[:200]
                 except Exception:
                      pass # Failed to get full info, description remains None
            
            # Estimate size from name and tags
            model_size_mb = ModelSizeEstimator.estimate_from_name_tags(model_id, tags)

            info = ModelHubInfo(
                model_id=model_id,
                author=author,
                model_name=model_name,
                downloads=getattr(model_info_obj, 'downloads', 0) or 0,
                likes=getattr(model_info_obj, 'likes', 0) or 0,
                tags=tags,
                pipeline_tag=getattr(model_info_obj, 'pipeline_tag', None),
                library_name=getattr(model_info_obj, 'library_name', None),
                model_size_mb=model_size_mb,
                description=description,
                last_modified=str(getattr(model_info_obj, 'lastModified', ''))
            )
            return info
            
        except Exception as e:
            logger.warning(f"Failed to process ModelInfo object for '{getattr(model_info_obj, 'modelId', 'unknown')}': {e}")
            return None


# --- Model Hub Class (Refactored) ---
class ModelHub:
    """Interface to HuggingFace Hub using refactored helpers."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.base_cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.model_cache_dir = self.base_cache_dir # Use the base cache dir directly
        self.info_cache_dir = self.base_cache_dir / "info"
        self.logger = get_logger_conf(__name__)
        
        self.api = HfApi()
        self.cache_manager = CacheManager(self.info_cache_dir / "hub_info_cache.json")
        self._model_cache: Dict[str, ModelHubInfo] = self._load_cache_from_manager()
        
        logger.info(f"Initialized ModelHub (Cache: {self.info_cache_dir})")
    
    def _load_cache_from_manager(self) -> Dict[str, ModelHubInfo]:
        """Load ModelHubInfo objects from cache file."""
        cache_data = self.cache_manager.load_cache()
        models = {}
        for model_id, data in cache_data.items():
            try:
                models[model_id] = ModelHubInfo(**data)
            except Exception:
                logger.warning(f"Removing invalid cache entry for {model_id}")
        logger.debug(f"Loaded {len(models)} models from hub info cache")
        return models

    def _save_cache(self):
        """Save current ModelHubInfo cache to disk."""
        # Convert ModelHubInfo objects back to plain dicts for JSON
        cache_data = {
            model_id: asdict(info)
            for model_id, info in self._model_cache.items()
        }
        self.cache_manager.save_cache(cache_data)

    @handle_hub_errors
    def search_models(self,
                     search_term: Optional[str] = None,
                     task: Optional[str] = "text-generation",
                     library: Optional[str] = "transformers",
                     sort: str = "downloads",
                     limit: int = 50,
                     min_downloads: int = 100) -> List[ModelHubInfo]:
        """Search models on HuggingFace Hub, applying local filters."""
        logger.info(f"Searching Hub: task={task}, library={library}, term='{search_term}'")
        
        # list_models returns an iterator
        # In newer versions of huggingface_hub, pass parameters directly
        model_iterator = self.api.list_models(
            task=task,
            library=library,
            search=search_term if search_term else None,
            sort=sort,
            direction=-1, # Descending
            fetch_config=False # Fetching config/cardData is slow, do it only if needed
        )
        
        model_infos = []
        try:
            for model in model_iterator:
                # Get minimal info from the iterator object
                info = ModelInfoExtractor.extract(model, api=None) # Don't fetch full info here
                if not info: continue
                
                # Apply local filters (downloads, etc.)
                if info.matches_filter(search_term=None, min_downloads=min_downloads): # search_term already applied server-side
                    model_infos.append(info)
                    self._model_cache[info.model_id] = info # Update cache
                
                if len(model_infos) >= limit:
                    break
        except Exception as e:
            # Catch errors during iteration (e.g., network issues)
            logger.error(f"Error during Hub model iteration: {e}", exc_info=True)
            ConsoleOutput.error(f"Error searching Hub: {e}")

        # Save cache after search
        self._save_cache()
        
        logger.info(f"Found {len(model_infos)} models matching criteria.")
        return model_infos
    
    @handle_hub_errors
    def get_model_info(self, model_id: str) -> Optional[ModelHubInfo]:
        """Get detailed info for a *single* model, updating the cache."""
        # Check cache first
        if model_id in self._model_cache:
            # Optionally refresh if cache is old? For now, just return
            return self._model_cache[model_id]
        
        logger.debug(f"Fetching full model info for {model_id} from Hub...")
        hf_api_info = self.api.model_info(model_id, files_metadata=True) # Get file info too if possible
        
        info = ModelInfoExtractor.extract(hf_api_info, api=self.api) # Pass API to allow description fetch
        
        if info:
            # Try a slightly better size estimate if files_metadata was fetched
            if hf_api_info.siblings:
                 total_size_bytes = sum(f.size for f in hf_api_info.siblings if f.rfilename.endswith(('.bin', '.safetensors', '.pt')))
                 if total_size_bytes > 0:
                      info.model_size_mb = total_size_bytes / (1024 * 1024)

            self._model_cache[model_id] = info
            self._save_cache()
            return info
        
        logger.warning(f"Could not retrieve or parse info for {model_id}")
        return None

    @handle_hub_errors
    @log_execution_time()
    def download_model(self,
                      model_id: str,
                      revision: str = "main",
                      allow_patterns: Optional[List[str]] = None,
                      ignore_patterns: Optional[List[str]] = None) -> Optional[Path]:
        """
        Download a model snapshot (or specific files) from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID
            revision: Model revision/branch
            allow_patterns: List of patterns to include (e.g., "*.json", "*.safetensors")
            ignore_patterns: List of patterns to exclude (e.g., "*.pt", "optimizer.bin")

        Returns:
            Path to downloaded model directory or None if failed
        """
        # Use a specific subdirectory within the main cache for HF models
        model_cache_path = self.model_cache_dir
        model_cache_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model: {model_id} (revision: {revision})")
        ConsoleOutput.info(f"Downloading {model_id}...")
        
        # Use snapshot_download to get the whole model directory
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(model_cache_path),
            revision=revision,
            resume_download=True,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            # token=HfFolder.get_token() # Use token if needed for gated models
        )
        
        logger.info(f"Model downloaded to: {model_path}")
        ConsoleOutput.success(f"Downloaded: {model_id} to {model_path}")
        
        # Mark as cached in registry
        get_registry().mark_cached(model_id, Path(model_path))
        
        return Path(model_path)
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model snapshot exists in the cache by checking for the directory."""
        model_path_name = "models--" + model_id.replace("/", "--")
        expected_path = self.model_cache_dir / model_path_name

        if expected_path.exists() and expected_path.is_dir():
            self.logger.debug(f"Found cached model {model_id} at {expected_path}")
            get_registry().mark_cached(model_id, expected_path)
            return True
        
        self.logger.debug(f"Model {model_id} not found at expected path: {expected_path}")
        return False


# --- Interactive Model Browser (Placeholder) ---

class InteractiveModelBrowser:
    """Placeholder for interactive model browsing functionality."""

    def __init__(self, model_hub: ModelHub):
        """Initialize browser with a ModelHub instance."""
        self.model_hub = model_hub
        self.logger = get_logger_conf(f"{__name__}.Browser")

    def browse(self):
        """Launch interactive model browsing."""
        self.logger.warning("InteractiveModelBrowser.browse() is not yet implemented.")
        ConsoleOutput.warning("Interactive model browsing is not yet implemented.")
        return None
