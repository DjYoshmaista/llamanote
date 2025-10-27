"""
Model Hub Module
Browse, search, and download models from HuggingFace Hub
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import json
from huggingface_hub import HfApi, hf_hub_download, snapshot_download, list_models
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
import requests
from loggerConf import get_logger_conf, ConsoleOutput
from config import CACHE_DIR

logger = get_logger_conf(__name__)


@dataclass
class ModelInfo:
    """Information about a HuggingFace model"""
    model_id: str
    author: str
    model_name: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str]
    library_name: Optional[str]
    model_size_mb: Optional[float]
    description: Optional[str]
    last_modified: Optional[str]
    
    @property
    def display_name(self) -> str:
        """Human-readable display name"""
        return f"{self.author}/{self.model_name} ({self.format_size()})"
    
    def format_size(self) -> str:
        """Format model size for display"""
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
        """Check if model matches filter criteria"""
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
            if not any(tag in self.tags for tag in tags):
                return False
        
        return True


class ModelHub:
    """Interface to HuggingFace Hub for browsing and downloading models"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize Model Hub
        
        Args:
            cache_dir: Directory for caching model metadata
        """
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.api = HfApi()
        self.cache_file = self.cache_dir / "model_cache.json"
        self._model_cache: Dict[str, ModelInfo] = {}
        self._load_cache()
        
        logger.info("Initialized ModelHub")
    
    def _load_cache(self):
        """Load cached model information"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for model_id, data in cache_data.items():
                    self._model_cache[model_id] = ModelInfo(**data)
                
                logger.debug(f"Loaded {len(self._model_cache)} models from cache")
            except Exception as e:
                logger.warning(f"Failed to load model cache: {e}")
    
    def _save_cache(self):
        """Save model cache to disk"""
        try:
            cache_data = {
                model_id: {
                    'model_id': info.model_id,
                    'author': info.author,
                    'model_name': info.model_name,
                    'downloads': info.downloads,
                    'likes': info.likes,
                    'tags': info.tags,
                    'pipeline_tag': info.pipeline_tag,
                    'library_name': info.library_name,
                    'model_size_mb': info.model_size_mb,
                    'description': info.description,
                    'last_modified': info.last_modified
                }
                for model_id, info in self._model_cache.items()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"Saved {len(self._model_cache)} models to cache")
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")
    
    def search_models(self,
                     search_term: Optional[str] = None,
                     task: Optional[str] = "text-generation",
                     library: str = "transformers",
                     sort: str = "downloads",
                     limit: int = 100,
                     min_downloads: int = 100) -> List[ModelInfo]:
        """
        Search for models on HuggingFace Hub
        
        Args:
            search_term: Search query (model name, description)
            task: Model task/pipeline (text-generation, text2text-generation, etc.)
            library: Library name (transformers, pytorch, etc.)
            sort: Sort by 'downloads', 'likes', or 'modified'
            limit: Maximum number of results
            min_downloads: Minimum download count filter
            
        Returns:
            List of ModelInfo objects
        """
        logger.info(f"Searching models: task={task}, library={library}, term='{search_term}'")
        
        try:
            # Search on HuggingFace Hub
            models = list_models(
                filter=task,
                library=library,
                sort=sort,
                direction=-1,  # Descending
                limit=limit * 2  # Get extra to account for filtering
            )
            
            model_infos = []
            
            for model in models:
                try:
                    # Extract model info
                    model_id = model.modelId
                    parts = model_id.split('/')
                    author = parts[0] if len(parts) > 1 else "unknown"
                    model_name = parts[-1]
                    
                    # Get additional info
                    downloads = getattr(model, 'downloads', 0) or 0
                    likes = getattr(model, 'likes', 0) or 0
                    tags = getattr(model, 'tags', []) or []
                    pipeline_tag = getattr(model, 'pipeline_tag', None)
                    library_name = getattr(model, 'library_name', None)
                    last_modified = str(getattr(model, 'lastModified', ''))
                    
                    # Try to get model card for description
                    description = None
                    try:
                        card_data = self.api.model_info(model_id)
                        if hasattr(card_data, 'cardData') and card_data.cardData:
                            description = str(card_data.cardData.get('description', ''))[:200]
                    except:
                        pass
                    
                    # Estimate model size (this is approximate)
                    model_size_mb = self._estimate_model_size(model_id, tags)
                    
                    info = ModelInfo(
                        model_id=model_id,
                        author=author,
                        model_name=model_name,
                        downloads=downloads,
                        likes=likes,
                        tags=tags,
                        pipeline_tag=pipeline_tag,
                        library_name=library_name,
                        model_size_mb=model_size_mb,
                        description=description,
                        last_modified=last_modified
                    )
                    
                    # Apply filters
                    if info.matches_filter(search_term, min_downloads):
                        model_infos.append(info)
                        self._model_cache[model_id] = info
                    
                    if len(model_infos) >= limit:
                        break
                        
                except Exception as e:
                    logger.debug(f"Failed to process model {model.modelId}: {e}")
                    continue
            
            # Save updated cache
            self._save_cache()
            
            logger.info(f"Found {len(model_infos)} models")
            return model_infos
            
        except Exception as e:
            logger.error(f"Model search failed: {e}")
            return []
    
    def _estimate_model_size(self, model_id: str, tags: List[str]) -> Optional[float]:
        """Estimate model size from tags and model ID"""
        # Look for size indicators in tags or model name
        size_indicators = {
            '270m': 270,
            '1b': 1000,
            '1.5b': 1500,
            '3b': 3000,
            '4b': 4000,
            '7b': 7000,
            '13b': 13000,
            '30b': 30000,
            '65b': 65000,
            '70b': 70000,
        }
        
        model_id_lower = model_id.lower()
        for size_str, size_mb in size_indicators.items():
            if size_str in model_id_lower or any(size_str in tag.lower() for tag in tags):
                return size_mb
        
        # Try to get actual size from repo
        try:
            files = self.api.list_repo_files(model_id)
            total_size = 0
            
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.pt', '.pth')):
                    try:
                        # This is a rough estimate
                        # In practice, you'd need to check file info
                        pass
                    except:
                        pass
            
            return None if total_size == 0 else total_size / (1024 * 1024)
        except:
            return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            ModelInfo object or None if not found
        """
        # Check cache first
        if model_id in self._model_cache:
            return self._model_cache[model_id]
        
        try:
            model = self.api.model_info(model_id)
            
            parts = model_id.split('/')
            author = parts[0] if len(parts) > 1 else "unknown"
            model_name = parts[-1]
            
            downloads = getattr(model, 'downloads', 0) or 0
            likes = getattr(model, 'likes', 0) or 0
            tags = getattr(model, 'tags', []) or []
            pipeline_tag = getattr(model, 'pipeline_tag', None)
            library_name = getattr(model, 'library_name', None)
            last_modified = str(getattr(model, 'lastModified', ''))
            
            description = None
            if hasattr(model, 'cardData') and model.cardData:
                description = str(model.cardData.get('description', ''))[:200]
            
            model_size_mb = self._estimate_model_size(model_id, tags)
            
            info = ModelInfo(
                model_id=model_id,
                author=author,
                model_name=model_name,
                downloads=downloads,
                likes=likes,
                tags=tags,
                pipeline_tag=pipeline_tag,
                library_name=library_name,
                model_size_mb=model_size_mb,
                description=description,
                last_modified=last_modified
            )
            
            self._model_cache[model_id] = info
            self._save_cache()
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_id}: {e}")
            return None
    
    def download_model(self,
                      model_id: str,
                      cache_dir: Optional[Path] = None,
                      revision: str = "main") -> Optional[Path]:
        """
        Download a model from HuggingFace Hub
        
        Args:
            model_id: HuggingFace model ID
            cache_dir: Directory to cache the model
            revision: Model revision/branch
            
        Returns:
            Path to downloaded model or None if failed
        """
        cache_dir = cache_dir or self.cache_dir / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading model: {model_id}")
        ConsoleOutput.info(f"Downloading {model_id}...")
        
        try:
            model_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                revision=revision,
                resume_download=True
            )
            
            logger.info(f"Model downloaded to: {model_path}")
            ConsoleOutput.success(f"Downloaded: {model_id}")
            
            return Path(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            ConsoleOutput.error(f"Download failed: {e}")
            return None
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is already downloaded"""
        try:
            # Check if model files exist in cache
            model_cache_path = self.cache_dir / "models" / model_id.replace('/', '--')
            return model_cache_path.exists()
        except:
            return False

    def save_to_registry(self, model_infos: List[ModelInfo]) -> int:
            """
            Save search results to the model registry
            
            Args:
                model_infos: List of ModelInfo objects
                
            Returns:
                Number of models saved
            """
            from model_registry import get_registry
            
            registry = get_registry()
            count = registry.update_from_hub(model_infos)
            
            logger.info(f"Saved {count} models to registry")
            return count

class InteractiveModelBrowser:
    """Interactive terminal interface for browsing and selecting models"""
    
    def __init__(self, model_hub: Optional[ModelHub] = None):
        self.hub = model_hub or ModelHub()
        self.current_results: List[ModelInfo] = []
        self.selected_model: Optional[ModelInfo] = None
    

    def browse(self, task: str = "text-generation", library: str = "transformers") -> Optional[ModelInfo]:
        """Interactive model browsing session"""
        ConsoleOutput.header("HuggingFace Model Browser")
        
        while True:
            print("\nOptions:")
            print("1. Search models")
            print("2. View popular models")
            print("3. Enter model ID directly")
            print("4. Update local registry from search")
            print("5. Back/Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                search_term = input("Enter search term: ").strip()
                self._search_and_display(search_term, task, library)
                
                if self.current_results:
                    model = self._select_from_results()
                    if model:
                        return model
            
            elif choice == "2":
                self._search_and_display(None, task, library)
                
                if self.current_results:
                    model = self._select_from_results()
                    if model:
                        return model
            
            elif choice == "3":
                model_id = input("Enter model ID: ").strip()
                if model_id:
                    info = self.hub.get_model_info(model_id)
                    if info:
                        ConsoleOutput.success(f"Found: {info.display_name}")
                        if self._confirm_selection(info):
                            return info
                    else:
                        ConsoleOutput.error("Model not found")
            
            elif choice == "4":
                if self.current_results:
                    confirm = input(f"Save {len(self.current_results)} models to registry? (y/n): ").strip().lower()
                    if confirm == 'y':
                        count = self.hub.save_to_registry(self.current_results)
                        ConsoleOutput.success(f"Saved {count} models to registry")
                else:
                    ConsoleOutput.warning("No search results to save")
            
            elif choice == "5":
                return None
            
            else:
                ConsoleOutput.warning("Invalid option")

    def _search_and_display(self, search_term: Optional[str], task: str, library: str):
        """Search and display results"""
        ConsoleOutput.section("Searching models...")
        
        self.current_results = self.hub.search_models(
            search_term=search_term,
            task=task,
            library=library,
            limit=20,
            min_downloads=1000
        )
        
        if not self.current_results:
            ConsoleOutput.warning("No models found")
            return
        
        ConsoleOutput.section(f"Found {len(self.current_results)} models")
        
        for i, model in enumerate(self.current_results, 1):
            cached = "✓" if self.hub.is_model_cached(model.model_id) else " "
            print(f"{i:2}. [{cached}] {model.display_name}")
            print(f"    Downloads: {model.downloads:,} | Likes: {model.likes}")
            if model.description:
                print(f"    {model.description[:80]}...")
    
    def _select_from_results(self) -> Optional[ModelInfo]:
        """Let user select from search results"""
        while True:
            choice = input(f"\nSelect model (1-{len(self.current_results)}) or 'b' for back: ").strip()
            
            if choice.lower() == 'b':
                return None
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.current_results):
                    model = self.current_results[idx]
                    
                    # Show detailed info
                    self._display_model_details(model)
                    
                    if self._confirm_selection(model):
                        return model
                else:
                    ConsoleOutput.warning("Invalid selection")
            except ValueError:
                ConsoleOutput.warning("Please enter a number")
    
    def _display_model_details(self, model: ModelInfo):
        """Display detailed model information"""
        ConsoleOutput.section(f"Model Details: {model.model_id}")
        print(f"Author: {model.author}")
        print(f"Size: {model.format_size()}")
        print(f"Downloads: {model.downloads:,}")
        print(f"Likes: {model.likes}")
        print(f"Pipeline: {model.pipeline_tag}")
        print(f"Library: {model.library_name}")
        print(f"Tags: {', '.join(model.tags[:10])}")
        print(f"Cached: {'Yes' if self.hub.is_model_cached(model.model_id) else 'No'}")
        
        if model.description:
            print(f"\nDescription:\n{model.description}")
    
    def _confirm_selection(self, model: ModelInfo) -> bool:
        """Confirm model selection"""
        confirm = input(f"\nSelect '{model.model_id}'? (y/n): ").strip().lower()
        return confirm == 'y'


def quick_model_search(search_term: str, limit: int = 10) -> List[ModelInfo]:
    """Quick search function for programmatic use"""
    hub = ModelHub()
    return hub.search_models(search_term=search_term, limit=limit)


def download_model_by_id(model_id: str) -> Optional[Path]:
    """Quick download function"""
    hub = ModelHub()
    return hub.download_model(model_id)
