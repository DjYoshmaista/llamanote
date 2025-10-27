"""
Dynamic Model Registry
Manages a local database of available models from HuggingFace
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import threading

from config_base import CACHE_DIR

# Thread-safe singleton registry
_registry_lock = threading.Lock()
_registry_instance = None


@dataclass
class ModelEntry:
    """Entry in the model registry"""
    model_id: str
    name: str
    author: str
    supports_thinking: bool = False
    thinking_tokens: List[str] = None
    max_context: int = 8192
    optimal_chunk_size: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: Optional[int] = None
    quantization_support: List[str] = None
    
    # Metadata from HuggingFace
    downloads: int = 0
    likes: int = 0
    tags: List[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    last_modified: Optional[str] = None
    
    # Local metadata
    is_cached: bool = False
    cache_path: Optional[str] = None
    added_date: Optional[str] = None
    is_predefined: bool = False
    
    def __post_init__(self):
        if self.thinking_tokens is None:
            self.thinking_tokens = []
        if self.quantization_support is None:
            self.quantization_support = ["4bit", "8bit"]
        if self.tags is None:
            self.tags = []
        if self.added_date is None:
            self.added_date = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelEntry':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_model_info(cls, model_info, is_predefined: bool = False) -> 'ModelEntry':
        """Create from ModelInfo object (from model_hub)"""
        return cls(
            model_id=model_info.model_id,
            name=model_info.model_name,
            author=model_info.author,
            downloads=model_info.downloads,
            likes=model_info.likes,
            tags=model_info.tags,
            pipeline_tag=model_info.pipeline_tag,
            library_name=model_info.library_name,
            last_modified=model_info.last_modified,
            is_predefined=is_predefined
        )


class ModelRegistry:
    """Central registry for all available models"""
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path or (CACHE_DIR / "model_registry.json")
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelEntry] = {}
        self._load_registry()
        self._ensure_predefined_models()
    
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    self.models[model_id] = ModelEntry.from_dict(model_data)
                
                print(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                print(f"Failed to load registry: {e}")
                self.models = {}
        else:
            self.models = {}
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {
                model_id: entry.to_dict()
                for model_id, entry in self.models.items()
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            print(f"Failed to save registry: {e}")
    
    def _ensure_predefined_models(self):
        """Ensure predefined models are in registry"""
        predefined = {
            "qwen3-4b": ModelEntry(
                model_id="Qwen/Qwen2.5-4B-Instruct",
                name="Qwen3-4B Thinking",
                author="Qwen",
                supports_thinking=True,
                thinking_tokens=["<think>", "</think>", "<|thinking|>", "<|/thinking|>"],
                max_context=32768,
                optimal_chunk_size=1500,
                temperature=0.7,
                top_p=0.9,
                is_predefined=True
            ),
            "gemma-270m": ModelEntry(
                model_id="google/gemma-2-2b-it",
                name="Gemma 2B",
                author="google",
                supports_thinking=False,
                max_context=8192,
                optimal_chunk_size=1000,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=2048,
                is_predefined=True
            ),
            "llama-3.2-1b": ModelEntry(
                model_id="meta-llama/Llama-3.2-1B-Instruct",
                name="Llama 3.2 1B",
                author="meta-llama",
                supports_thinking=False,
                max_context=8192,
                optimal_chunk_size=1200,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=2048,
                is_predefined=True
            ),
        }
        
        # Add predefined models if they don't exist
        updated = False
        for key, entry in predefined.items():
            if entry.model_id not in self.models:
                self.models[entry.model_id] = entry
                updated = True
        
        if updated:
            self._save_registry()
    
    def add_model(self, entry: ModelEntry, save: bool = True) -> bool:
        """
        Add a model to the registry
        
        Args:
            entry: ModelEntry to add
            save: Whether to save registry to disk
            
        Returns:
            True if added, False if already exists
        """
        if entry.model_id in self.models:
            # Update existing entry
            self.models[entry.model_id] = entry
        else:
            self.models[entry.model_id] = entry
        
        if save:
            self._save_registry()
        
        return True
    
    def add_from_model_info(self, model_info, save: bool = True) -> bool:
        """Add model from ModelInfo object"""
        entry = ModelEntry.from_model_info(model_info)
        return self.add_model(entry, save)
    
    def add_batch(self, entries: List[ModelEntry]):
        """Add multiple models at once"""
        for entry in entries:
            self.models[entry.model_id] = entry
        
        self._save_registry()
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """Get a model by ID"""
        # Try exact match
        if model_id in self.models:
            return self.models[model_id]
        
        # Try partial match
        for mid, entry in self.models.items():
            if model_id.lower() in mid.lower():
                return entry
        
        return None
    
    def get_by_key(self, key: str) -> Optional[ModelEntry]:
        """Get predefined model by short key (e.g., 'qwen3-4b')"""
        # Check if this is a predefined key
        for model_id, entry in self.models.items():
            if entry.is_predefined and key in model_id.lower():
                return entry
        
        return None
    
    def list_models(self, 
                   predefined_only: bool = False,
                   cached_only: bool = False,
                   sort_by: str = "downloads") -> List[ModelEntry]:
        """
        List models in registry
        
        Args:
            predefined_only: Only return predefined models
            cached_only: Only return cached models
            sort_by: Sort by 'downloads', 'likes', 'name', or 'date'
            
        Returns:
            List of ModelEntry objects
        """
        models = list(self.models.values())
        
        if predefined_only:
            models = [m for m in models if m.is_predefined]
        
        if cached_only:
            models = [m for m in models if m.is_cached]
        
        # Sort
        if sort_by == "downloads":
            models.sort(key=lambda m: m.downloads, reverse=True)
        elif sort_by == "likes":
            models.sort(key=lambda m: m.likes, reverse=True)
        elif sort_by == "name":
            models.sort(key=lambda m: m.name.lower())
        elif sort_by == "date":
            models.sort(key=lambda m: m.added_date or "", reverse=True)
        
        return models
    
    def search(self, query: str) -> List[ModelEntry]:
        """Search models by query string"""
        query_lower = query.lower()
        results = []
        
        for entry in self.models.values():
            if (query_lower in entry.model_id.lower() or
                query_lower in entry.name.lower() or
                query_lower in entry.author.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)):
                results.append(entry)
        
        return results
    
    def update_from_hub(self, model_infos: List) -> int:
        """
        Update registry from HuggingFace model search results
        
        Args:
            model_infos: List of ModelInfo objects from model_hub
            
        Returns:
            Number of models added/updated
        """
        count = 0
        entries = []
        
        for info in model_infos:
            entry = ModelEntry.from_model_info(info)
            entries.append(entry)
            count += 1
        
        self.add_batch(entries)
        return count
    
    def mark_cached(self, model_id: str, cache_path: Path):
        """Mark a model as cached"""
        if model_id in self.models:
            self.models[model_id].is_cached = True
            self.models[model_id].cache_path = str(cache_path)
            self._save_registry()
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from registry"""
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            return True
        return False
    
    def clear_non_predefined(self):
        """Remove all non-predefined models"""
        self.models = {
            mid: entry for mid, entry in self.models.items()
            if entry.is_predefined
        }
        self._save_registry()
    
    def export_to_json(self, path: Path):
        """Export registry to a JSON file"""
        data = {
            model_id: entry.to_dict()
            for model_id, entry in self.models.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_from_json(self, path: Path):
        """Import registry from a JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for model_id, model_data in data.items():
            entry = ModelEntry.from_dict(model_data)
            self.models[model_id] = entry
        
        self._save_registry()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        models = list(self.models.values())
        
        return {
            "total_models": len(models),
            "predefined_models": sum(1 for m in models if m.is_predefined),
            "cached_models": sum(1 for m in models if m.is_cached),
            "thinking_models": sum(1 for m in models if m.supports_thinking),
            "total_downloads": sum(m.downloads for m in models),
            "total_likes": sum(m.likes for m in models),
        }


def get_registry() -> ModelRegistry:
    """Get the global model registry instance (singleton)"""
    global _registry_instance
    
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ModelRegistry()
    
    return _registry_instance


def refresh_registry_from_hub(search_terms: List[str] = None, limit: int = 100) -> int:
    """
    Refresh registry by searching HuggingFace
    
    Args:
        search_terms: List of search terms (None = get popular models)
        limit: Number of models to fetch per search term
        
    Returns:
        Number of models added/updated
    """
    from model_hub import ModelHub
    
    hub = ModelHub()
    registry = get_registry()
    
    if search_terms is None:
        search_terms = ["instruct", "chat", "llm"]
    
    all_models = []
    
    for term in search_terms:
        models = hub.search_models(
            search_term=term,
            task="text-generation",
            library="transformers",
            limit=limit,
            min_downloads=1000
        )
        all_models.extend(models)
    
    # Remove duplicates
    seen = set()
    unique_models = []
    for model in all_models:
        if model.model_id not in seen:
            seen.add(model.model_id)
            unique_models.append(model)
    
    count = registry.update_from_hub(unique_models)
    
    return count


# Convenience functions for backward compatibility
def get_model_config(model_identifier: str) -> Optional[ModelEntry]:
    """Get model config by ID or key"""
    registry = get_registry()
    
    # Try as direct ID
    model = registry.get_model(model_identifier)
    if model:
        return model
    
    # Try as predefined key
    model = registry.get_by_key(model_identifier)
    return model


def list_available_models(predefined_only: bool = False) -> List[ModelEntry]:
    """List all available models"""
    registry = get_registry()
    return registry.list_models(predefined_only=predefined_only)
