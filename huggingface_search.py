"""
HuggingFace Model Search Module
Search, filter, and manage TTS models from HuggingFace
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from datetime import datetime

from logging_config import get_logger, ConsoleOutput
from config_manager import ConfigManager

logger = get_logger(__name__)


class ModelTask(Enum):
    """HuggingFace model tasks"""
    TEXT_TO_SPEECH = "text-to-speech"
    TEXT_TO_AUDIO = "text-to-audio"
    AUDIO_TO_AUDIO = "audio-to-audio"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"


@dataclass
class ModelInfo:
    """Information about a HuggingFace model"""
    model_id: str
    task: str
    downloads: int
    likes: int
    created_at: str
    last_modified: str
    tags: List[str]
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    language: Optional[List[str]]
    license: Optional[str]
    dataset: Optional[List[str]]
    metrics: Optional[Dict[str, Any]]
    
    @property
    def display_name(self) -> str:
        """Get display name for model"""
        return self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id


@dataclass
class SearchFilters:
    """Filters for model search"""
    task: Optional[ModelTask] = ModelTask.TEXT_TO_SPEECH
    min_downloads: int = 0
    min_likes: int = 0
    languages: List[str] = None
    libraries: List[str] = None
    licenses: List[str] = None
    tags: List[str] = None
    author: Optional[str] = None
    search_query: Optional[str] = None
    sort_by: str = "downloads"  # downloads, likes, created, modified
    limit: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API"""
        params = {}
        
        if self.task:
            params["pipeline_tag"] = self.task.value
            
        if self.search_query:
            params["search"] = self.search_query
            
        if self.author:
            params["author"] = self.author
            
        if self.limit:
            params["limit"] = self.limit
            
        params["sort"] = self.sort_by
        params["direction"] = -1  # Descending
        
        return params
        
    def matches(self, model: ModelInfo) -> bool:
        """Check if model matches filters"""
        # Check downloads
        if model.downloads < self.min_downloads:
            return False
            
        # Check likes
        if model.likes < self.min_likes:
            return False
            
        # Check languages
        if self.languages and model.language:
            if not any(lang in model.language for lang in self.languages):
                return False
                
        # Check libraries
        if self.libraries and model.library_name:
            if model.library_name not in self.libraries:
                return False
                
        # Check licenses
        if self.licenses and model.license:
            if model.license not in self.licenses:
                return False
                
        # Check tags
        if self.tags:
            if not any(tag in model.tags for tag in self.tags):
                return False
                
        return True


class HuggingFaceSearch:
    """Search and manage HuggingFace models"""
    
    API_URL = "https://huggingface.co/api/models"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize HuggingFace search
        
        Args:
            cache_dir: Directory for caching search results
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "llamanote" / "hf_search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.config_manager = ConfigManager()
        
        logger.info("Initialized HuggingFaceSearch")
        
    def search_models(self, 
                     filters: Optional[SearchFilters] = None,
                     use_cache: bool = True) -> List[ModelInfo]:
        """
        Search for models on HuggingFace
        
        Args:
            filters: Search filters
            use_cache: Whether to use cached results
            
        Returns:
            List of matching models
        """
        filters = filters or SearchFilters()
        
        # Check cache
        cache_key = self._get_cache_key(filters)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                logger.info(f"Using cached results for {cache_key}")
                return cached
                
        logger.info("Searching HuggingFace models...")
        
        try:
            # Make API request
            params = filters.to_dict()
            response = self.session.get(self.API_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            models_data = response.json()
            
            # Convert to ModelInfo objects
            models = []
            for data in models_data:
                try:
                    model = self._parse_model_data(data)
                    if model and filters.matches(model):
                        models.append(model)
                except Exception as e:
                    logger.debug(f"Failed to parse model data: {e}")
                    continue
                    
            logger.info(f"Found {len(models)} models matching filters")
            
            # Cache results
            if use_cache:
                self._save_cache(cache_key, models)
                
            return models
            
        except requests.RequestException as e:
            logger.error(f"Failed to search models: {e}")
            return []
            
    def _parse_model_data(self, data: Dict[str, Any]) -> Optional[ModelInfo]:
        """Parse model data from API response"""
        try:
            return ModelInfo(
                model_id=data.get("modelId", data.get("id", "")),
                task=data.get("pipeline_tag", ""),
                downloads=data.get("downloads", 0),
                likes=data.get("likes", 0),
                created_at=data.get("createdAt", ""),
                last_modified=data.get("lastModified", ""),
                tags=data.get("tags", []),
                library_name=data.get("library_name"),
                pipeline_tag=data.get("pipeline_tag"),
                language=data.get("language"),
                license=data.get("license"),
                dataset=data.get("dataset"),
                metrics=data.get("metrics")
            )
        except Exception as e:
            logger.debug(f"Failed to create ModelInfo: {e}")
            return None
            
    def _get_cache_key(self, filters: SearchFilters) -> str:
        """Generate cache key for filters"""
        # Create a hash of filter parameters
        import hashlib
        filter_str = json.dumps(filters.to_dict(), sort_keys=True)
        return hashlib.md5(filter_str.encode()).hexdigest()[:12]
        
    def _load_cache(self, cache_key: str) -> Optional[List[ModelInfo]]:
        """Load cached search results"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        # Check cache age (expire after 24 hours)
        age_hours = (datetime.now().timestamp() - cache_file.stat().st_mtime) / 3600
        if age_hours > 24:
            logger.debug(f"Cache expired for {cache_key}")
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            models = []
            for model_data in data:
                model = ModelInfo(**model_data)
                models.append(model)
                
            return models
            
        except Exception as e:
            logger.debug(f"Failed to load cache: {e}")
            return None
            
    def _save_cache(self, cache_key: str, models: List[ModelInfo]):
        """Save search results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            data = [asdict(model) for model in models]
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"Cached {len(models)} models to {cache_key}")
            
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
            
    def get_popular_tts_models(self, limit: int = 10) -> List[ModelInfo]:
        """Get popular TTS models"""
        filters = SearchFilters(
            task=ModelTask.TEXT_TO_SPEECH,
            min_downloads=1000,
            sort_by="downloads",
            limit=limit
        )
        
        return self.search_models(filters)
        
    def search_by_language(self, language: str, 
                          task: ModelTask = ModelTask.TEXT_TO_SPEECH) -> List[ModelInfo]:
        """Search models by language"""
        filters = SearchFilters(
            task=task,
            languages=[language],
            sort_by="downloads"
        )
        
        return self.search_models(filters)
        
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        try:
            url = f"https://huggingface.co/api/models/{model_id}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Failed to get model details: {e}")
            return None


class FilterBuilder:
    """Interactive filter builder for model search"""
    
    AVAILABLE_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", 
        "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi", "cs"
    ]
    
    AVAILABLE_LIBRARIES = [
        "transformers", "speechbrain", "fairseq", "espnet", 
        "coqui", "bark", "tortoise", "piper", "vall-e"
    ]
    
    COMMON_LICENSES = [
        "apache-2.0", "mit", "cc-by-4.0", "cc-by-nc-4.0", 
        "openrail", "gpl-3.0", "bsd-3-clause"
    ]
    
    def __init__(self):
        self.filters = SearchFilters()
        
    def interactive_build(self) -> SearchFilters:
        """Build filters interactively"""
        ConsoleOutput.section("Filter Configuration")
        
        # Task selection
        print("\nSelect task:")
        tasks = list(ModelTask)
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task.value}")
        
        choice = input("\nTask (1-{}, default=1): ".format(len(tasks))).strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(tasks):
                self.filters.task = tasks[idx]
                
        # Minimum downloads
        min_dl = input("\nMinimum downloads (default=0): ").strip()
        if min_dl and min_dl.isdigit():
            self.filters.min_downloads = int(min_dl)
            
        # Minimum likes
        min_likes = input("Minimum likes (default=0): ").strip()
        if min_likes and min_likes.isdigit():
            self.filters.min_likes = int(min_likes)
            
        # Languages
        print("\nAvailable languages:", ", ".join(self.AVAILABLE_LANGUAGES[:10]), "...")
        langs = input("Languages (comma-separated, e.g., 'en,es,fr'): ").strip()
        if langs:
            self.filters.languages = [l.strip() for l in langs.split(",")]
            
        # Libraries
        print("\nAvailable libraries:", ", ".join(self.AVAILABLE_LIBRARIES[:5]), "...")
        libs = input("Libraries (comma-separated, or press Enter to skip): ").strip()
        if libs:
            self.filters.libraries = [l.strip() for l in libs.split(",")]
            
        # Search query
        query = input("\nSearch query (optional): ").strip()
        if query:
            self.filters.search_query = query
            
        # Sort order
        print("\nSort by:")
        print("  1. Downloads (most popular)")
        print("  2. Likes")
        print("  3. Recently created")
        print("  4. Recently updated")
        
        sort_choice = input("Choice (1-4, default=1): ").strip()
        sort_map = {"1": "downloads", "2": "likes", "3": "created", "4": "modified"}
        self.filters.sort_by = sort_map.get(sort_choice, "downloads")
        
        # Result limit
        limit = input("\nMaximum results (default=100): ").strip()
        if limit and limit.isdigit():
            self.filters.limit = int(limit)
            
        return self.filters
        
    def from_preset(self, preset: str) -> SearchFilters:
        """Load filters from preset"""
        presets = {
            "popular": SearchFilters(
                min_downloads=10000,
                min_likes=50,
                sort_by="downloads"
            ),
            "english": SearchFilters(
                languages=["en"],
                min_downloads=1000
            ),
            "multilingual": SearchFilters(
                tags=["multilingual"],
                min_downloads=5000
            ),
            "recent": SearchFilters(
                sort_by="created",
                limit=50
            ),
            "bark": SearchFilters(
                libraries=["bark"],
                sort_by="downloads"
            ),
            "coqui": SearchFilters(
                libraries=["coqui"],
                sort_by="downloads"
            )
        }
        
        return presets.get(preset, SearchFilters())
        
    def save_filters(self, name: str, filters: SearchFilters):
        """Save filters to file"""
        config_manager = ConfigManager()
        config_manager.save_filter_preset(name, asdict(filters))
        
    def load_filters(self, name: str) -> Optional[SearchFilters]:
        """Load filters from file"""
        config_manager = ConfigManager()
        data = config_manager.load_filter_preset(name)
        
        if data:
            # Convert task string back to enum
            if 'task' in data and data['task']:
                data['task'] = ModelTask(data['task'])
            return SearchFilters(**data)
        
        return None


class ModelDownloader:
    """Download and cache HuggingFace models"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Directory for model cache
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_model(self, model_id: str, 
                      revision: str = "main",
                      force: bool = False) -> Optional[Path]:
        """
        Download model from HuggingFace
        
        Args:
            model_id: Model identifier
            revision: Model revision/branch
            force: Force re-download
            
        Returns:
            Path to downloaded model or None
        """
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model: {model_id}")
            
            local_dir = self.cache_dir / model_id.replace("/", "--")
            
            if local_dir.exists() and not force:
                logger.info(f"Model already cached at {local_dir}")
                return local_dir
                
            # Download model
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=self.cache_dir,
                local_dir=local_dir if force else None,
                local_dir_use_symlinks=False if force else True,
                resume_download=True
            )
            
            logger.info(f"Model downloaded to {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return None
            
    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is already cached"""
        local_dir = self.cache_dir / model_id.replace("/", "--")
        return local_dir.exists() and any(local_dir.iterdir())
        
    def get_cached_models(self) -> List[str]:
        """Get list of cached models"""
        models = []
        
        for path in self.cache_dir.iterdir():
            if path.is_dir() and "--" in path.name:
                model_id = path.name.replace("--", "/")
                models.append(model_id)
                
        return models
        
    def clear_cache(self, model_id: Optional[str] = None):
        """Clear model cache"""
        if model_id:
            local_dir = self.cache_dir / model_id.replace("/", "--")
            if local_dir.exists():
                import shutil
                shutil.rmtree(local_dir)
                logger.info(f"Cleared cache for {model_id}")
        else:
            # Clear all cache
            import shutil
            for path in self.cache_dir.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
            logger.info("Cleared all model cache")
