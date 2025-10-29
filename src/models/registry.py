# llamanote/models/registry.py
"""
Dynamic Model Registry
Manages a local database of available models from HuggingFace and local files.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
from typing import Dict, List, Optional, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .hub import ModelHubInfo
# Use config/settings.py for base paths
from ..config.settings import DEFAULT_CACHE_DIR, DEFAULT_MODEL_KEY, FALLBACK_MODEL_KEY
from ..utils.logger import get_logger_conf
from ..config.manager import _ensure_directory

logger = get_logger_conf(__name__)

# Thread-safe singleton registry
_registry_lock = threading.Lock()
_registry_instance = None

# Location of the predefined models file (relative to this file)
PREDEFINED_MODELS_FILE = Path(__file__).parent.parent / "config" / "predefined_models.json"

# --- Model Entry Dataclass ---

@dataclass
class ModelEntry:
    """Entry in the model registry"""
    model_id: str         # The unique identifier (e.g., "Qwen/Qwen3-4B-Instruct-2507" or path)
    name: str             # User-friendly display name
    author: str           # Model author/organization
    
    short_key: Optional[str] = None  # Optional short alias (e.g., "qwen3-4b")
    supports_thinking: bool = False  # Does it use <think> tags?
    thinking_tokens: List[str] = field(default_factory=list) # List of start/end tags
    
    # --- Performance Hints ---
    max_context: int = 8192         # Default context window size
    optimal_chunk_size: int = 1000  # Recommended chunk size
    quantization_support: List[str] = field(default_factory=lambda: ["4bit", "8bit"])

    # --- Default Hyperparameters (can be overridden) ---
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: Optional[int] = 2048 # Default max *new* tokens

    # --- Metadata (from Hub or local) ---
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = "transformers" # Default library
    last_modified: Optional[str] = None
    
    # --- Local State ---
    is_cached: bool = False
    cache_path: Optional[str] = None
    added_date: str = field(default_factory=lambda: datetime.now().isoformat())
    is_predefined: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelEntry':
        """Create ModelEntry from a dictionary (e.g., loaded from JSON)."""
        # Get all field names from the dataclass definition
        known_fields = cls.__annotations__.keys()
        # Filter data to only include keys that are fields in ModelEntry
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)
    
    @classmethod
    def from_model_info(cls, model_info: 'ModelHubInfo', is_predefined: bool = False) -> 'ModelEntry':
        """Create from ModelInfo object (from model_hub)."""
        # 'HFModelInfo' is the dataclass from models/hub.py
        # This assumes HFModelInfo has matching attribute names
        return cls(
            model_id=model_info.model_id,
            name=model_info.model_name,
            author=model_info.author,
            downloads=model_info.downloads or 0,
            likes=model_info.likes or 0,
            tags=model_info.tags or [],
            pipeline_tag=model_info.pipeline_tag,
            library_name=model_info.library_name or "transformers",
            last_modified=model_info.last_modified,
            is_predefined=is_predefined
            # Other fields (supports_thinking, max_context, etc.) use defaults
        )


class ModelRegistry:
    """Central registry for all available models."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or (CACHE_DIR / "model_registry.json")
        _ensure_directory(self.registry_path.parent)
        
        self.models: Dict[str, ModelEntry] = {}
        self._load_registry()
        self._ensure_predefined_models()
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    if "model_id" not in model_data: # Ensure key is present
                        model_data["model_id"] = model_id
                    try:
                        self.models[model_id] = ModelEntry.from_dict(model_data)
                    except TypeError as e:
                        logger.warning(f"Skipping incompatible model entry '{model_id}' in registry: {e}")
                
                logger.info(f"Loaded {len(self.models)} models from registry: {self.registry_path}")
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}", exc_info=True)
                self.models = {}
        else:
            logger.info("No existing model registry found. Starting fresh.")
            self.models = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                model_id: entry.to_dict()
                for model_id, entry in self.models.items()
            }
            
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.debug(f"Saved {len(self.models)} models to registry.")
            
        except Exception as e:
            logger.error(f"Failed to save registry to {self.registry_path}: {e}", exc_info=True)
    
    def _load_predefined_models_file(self) -> Dict[str, Dict[str, Any]]:
        """Loads predefined models from the external JSON file."""
        if not PREDEFINED_MODELS_FILE.exists():
            logger.warning(f"Predefined models file not found: {PREDEFINED_MODELS_FILE}. Using fallback defaults.")
            # Provide minimal hardcoded fallback
            return {
                DEFAULT_MODEL_KEY: {"model_id": "Qwen/Qwen3-4B-Instruct-2507", "name": "Qwen3-4B (Default)", "author": "Qwen", "supports_thinking": True, "max_context": 32768, "optimal_chunk_size": 2000},
                FALLBACK_MODEL_KEY: {"model_id": "google/gemma-3-270m", "name": "Gemma 3 270M (Fallback)", "author": "google", "max_context": 8192, "optimal_chunk_size": 1000},
            }
        try:
            with open(PREDEFINED_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load predefined models from {PREDEFINED_MODELS_FILE}: {e}", exc_info=True)
            return {} # Return empty on error to avoid overwriting registry with bad data

    def _ensure_predefined_models(self):
        """Ensures predefined models from file are present in the registry."""
        predefined_data = self._load_predefined_models_file()
        updated = False
        
        for key, data in predefined_data.items():
            try:
                # Ensure model_id is present, default from key if missing (unlikely)
                if 'model_id' not in data:
                     logger.warning(f"Predefined model '{key}' missing 'model_id'. Skipping.")
                     continue
                
                model_id = data['model_id']
                
                # Set predefined flags
                data['is_predefined'] = True
                data['short_key'] = key
                
                entry = ModelEntry.from_dict(data)
                
                # Add or update logic
                if model_id not in self.models or not self.models[model_id].is_predefined:
                     # Add if new, or overwrite if existing one wasn't marked predefined
                     self.models[model_id] = entry
                     updated = True
                else:
                     # If it exists and IS predefined, optionally update fields
                     # For now, just ensure short_key is set
                     if self.models[model_id].short_key != key:
                          self.models[model_id].short_key = key
                          updated = True
                          
            except Exception as e:
                 logger.warning(f"Skipping invalid predefined model entry '{key}': {e}")

        if updated:
            self._save_registry()
            
    def add_models(self, *entries: ModelEntry, save: bool = True):
        """Adds one or more ModelEntry objects to the registry, overwriting existing."""
        added_count = 0
        updated_count = 0
        for entry in entries:
            if not isinstance(entry, ModelEntry) or not entry.model_id:
                logger.warning(f"Skipping invalid model entry: {entry}")
                continue
            
            if entry.model_id in self.models:
                # Merge: Update existing entry with new info, keeping some old fields
                existing = self.models[entry.model_id]
                new_data = entry.to_dict()
                # Preserve local state like is_cached, cache_path, and is_predefined
                new_data['is_cached'] = existing.is_cached
                new_data['cache_path'] = existing.cache_path
                # Only overwrite is_predefined if the new entry explicitly sets it
                if 'is_predefined' not in new_data or not new_data['is_predefined']:
                    new_data['is_predefined'] = existing.is_predefined
                    new_data['short_key'] = existing.short_key
                
                self.models[entry.model_id] = ModelEntry.from_dict(new_data)
                updated_count += 1
            else:
                self.models[entry.model_id] = entry
                added_count += 1
        
        logger.debug(f"Registry: Added {added_count}, Updated {updated_count} model entries.")
        if save:
            self._save_registry()

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Gets a model by its full ID."""
        return self.models.get(model_id)

    def get_by_key(self, key: str) -> Optional[ModelEntry]:
        """Gets a predefined model by its short key (case-insensitive)."""
        key_lower = key.lower()
        for entry in self.models.values():
            if entry.is_predefined and entry.short_key and entry.short_key.lower() == key_lower:
                return entry
        logger.debug(f"Short key '{key}' not found in predefined models.")
        return None
    
    def find_entry(self, identifier: str) -> Optional[ModelEntry]:
        """
        Finds a model entry by short key first, then by full model_id.
        
        Args:
            identifier: A short key (e.g., "qwen3-4b") or full ID (e.g., "Qwen/Qwen3-4B-Instruct-2507")
            
        Returns:
            The matching ModelEntry, or None.
        """
        if not identifier: return None
        
        # 1. Try by short key
        if '/' not in identifier: # Assume it's a key if no slash
            entry = self.get_by_key(identifier)
            if entry:
                return entry
        
        # 2. Try by full model_id
        entry = self.get_model(identifier)
        if entry:
            return entry
            
        # 3. Try partial match as fallback? (Maybe too ambiguous)
        # ...
        
        logger.debug(f"Model identifier '{identifier}' not found in registry.")
        return None


    def list_models(self, **filters) -> List[ModelEntry]:
        """
        Lists models, optionally filtered by criteria and sorted.
        
        Args:
            **filters: Key-value pairs to filter on (e.g., is_predefined=True, author="google")
            sort_by: (str, optional) Key to sort by (e.g., "downloads", "name"). Defaults to "name".
            
        Returns:
            List of ModelEntry objects
        """
        sort_key = filters.pop('sort_by', 'name')
        
        results = []
        for entry in self.models.values():
            match = True
            for key, value in filters.items():
                if not hasattr(entry, key) or getattr(entry, key) != value:
                    match = False
                    break
            if match:
                results.append(entry)
                
        # Sort results
        reverse_sort = sort_key in ["downloads", "likes", "added_date"]
        try:
             results.sort(key=lambda m: getattr(m, sort_key) or 0, reverse=reverse_sort)
        except TypeError:
             # Fallback for mixed types (e.g., None vs. str)
             results.sort(key=lambda m: str(getattr(m, sort_key, "")), reverse=reverse_sort)
        except AttributeError:
             logger.warning(f"Cannot sort by '{sort_key}', sorting by name instead.")
             results.sort(key=lambda m: m.name.lower())
             
        return models
    
    def update_from_hub_list(self, model_infos: List['ModelHubInfo']) -> int:
        """Updates the registry from a list of ModelInfo objects."""
        if not model_infos: return 0
        
        entries = [ModelEntry.from_model_info(info) for info in model_infos]
        self.add_batch(entries, save=True) # Save after adding all
        return len(entries)

    def mark_cached(self, model_id: str, cache_path: Path, save: bool = True):
        """Mark a model as cached."""
        if model_id in self.models:
            self.models[model_id].is_cached = True
            self.models[model_id].cache_path = str(cache_path.resolve())
            if save: self._save_registry()
        else:
             logger.warning(f"Tried to mark non-existent model '{model_id}' as cached.")
    
    # ... (remove_model, clear_non_predefined, export, import, get_statistics as before) ...
    # ... (These methods are largely compatible with the ConfigCRUD refactoring) ...
    # ... (Small adjustment needed for export/import to use the new from_dict/to_dict) ...
    
    def export_to_json(self, path: Path):
        """Export registry to a JSON file."""
        try:
            data = { model_id: entry.to_dict() for model_id, entry in self.models.items() }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            logger.info(f"Exported {len(data)} models to {path}")
        except Exception as e:
            logger.error(f"Failed to export registry to {path}: {e}", exc_info=True)

    def import_from_json(self, path: Path, overwrite_predefined: bool = False):
        """Import registry from a JSON file, merging with existing."""
        if not path.is_file():
            logger.error(f"Import file not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            updated_count = 0
            for model_id, model_data in data.items():
                if not isinstance(model_data, dict):
                    logger.warning(f"Skipping invalid data entry for key '{model_id}' during import.")
                    continue
                
                # Ensure model_id is in the data dict for from_dict
                model_data.setdefault('model_id', model_id) 
                
                # Check if it already exists
                if model_id in self.models:
                    if self.models[model_id].is_predefined and not overwrite_predefined:
                         logger.debug(f"Skipping import for predefined model (overwrite=False): {model_id}")
                         continue
                    updated_count += 1
                else:
                    added_count += 1
                    
                try:
                     entry = ModelEntry.from_dict(model_data)
                     self.models[model_id] = entry # Add or overwrite
                except Exception as e:
                     logger.warning(f"Failed to import model entry '{model_id}': {e}")
            
            self._save_registry()
            logger.info(f"Import complete: Added {added_count}, Updated {updated_count} models from {path}")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to import registry: Invalid JSON file at {path}")
        except Exception as e:
            logger.error(f"Failed to import registry: {e}", exc_info=True)


    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = list(self.models.values())
        
        return {
            "total_models": len(models),
            "predefined_models": sum(1 for m in models if m.is_predefined),
            "cached_models": sum(1 for m in models if m.is_cached),
            "thinking_models": sum(1 for m in models if m.supports_thinking),
            "total_downloads": sum(m.downloads for m in models if m.downloads),
            "total_likes": sum(m.likes for m in models if m.likes),
        }


# --- Singleton Access ---
def get_registry() -> ModelRegistry:
    """Get the global model registry instance (singleton)"""
    global _registry_instance
    
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                # Uses the default path (in config_base.py via settings.py)
                _registry_instance = ModelRegistry() 
    
    return _registry_instance


# --- Module-level utility functions ---

def get_model_entry(identifier: str) -> Optional[ModelEntry]:
     """Convenience function to get ModelEntry by short key or full ID."""
     registry = get_registry()
     return registry.find_entry(identifier)


def list_available_models(predefined_only: bool = False, **filters) -> List[ModelEntry]:
    """List all available models, with optional filters."""
    registry = get_registry()
    filters['predefined_only'] = predefined_only
    return registry.list_models(**filters)


def refresh_registry_from_hub(search_terms: List[str] = None, limit: int = 50) -> int:
    """
    Refresh registry by searching HuggingFace.
    (Moved from llamanote.py)
    """
    from .hub import ModelHub # Local import to avoid circular dependency
    
    hub = ModelHub(cache_dir=CACHE_DIR / "model_hub") # Use specific cache
    registry = get_registry()
    
    if search_terms is None:
        search_terms = ["instruct", "chat"] # Default search terms
    
    logger.info(f"Refreshing registry from Hub with terms: {search_terms} (limit {limit} each)...")
    
    all_models_info = []
    seen_ids = set(registry.models.keys()) # Avoid re-fetching full info if already in registry
    
    for term in search_terms:
        try:
            models = hub.search_models(
                search_term=term,
                task="text-generation",
                library="transformers",
                limit=limit,
                min_downloads=500 # Set a minimum bar for relevance
            )
            for info in models:
                 if info.model_id not in seen:
                      all_models_info.append(info)
                      seen.add(info.model_id)
        except Exception as e:
            logger.error(f"Error searching hub for term '{term}': {e}")
            
    if not all_models_info:
        logger.info("No new models found on Hub matching criteria.")
        return 0
    
    # Update registry
    count = registry.update_from_hub_list(all_models_info)
    logger.info(f"Registry updated. Added/updated {count} models from Hub.")
    return count
