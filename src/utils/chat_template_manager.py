"""
Chat Template Manager - Modular Template Management System

This module provides a self-contained, reusable system for managing chat templates
for LLM tokenizers. It can be used standalone or integrated into larger systems.

Key Features:
- Automatic template detection and application
- Fallback templates for models without built-in templates
- Support for custom templates
- Template validation and testing
- Multiple template formats (ChatML, Llama, Alpaca, etc.)

Dependencies:
- Standard library only (json, re, logging)
- Optional: transformers (for tokenizer integration)

Usage:
    # Standalone usage
    from chat_template_manager import ChatTemplateManager

    manager = ChatTemplateManager()
    template = manager.get_template("TinyLlama/TinyLlama_v1.1")
    manager.apply_template(tokenizer, "TinyLlama/TinyLlama_v1.1")

    # With custom template
    manager.register_template("my-model", custom_template_string)
    manager.apply_template(tokenizer, "my-model")
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Template Definitions
# ============================================================================

@dataclass
class ChatTemplate:
    """Represents a chat template configuration."""
    name: str
    template: str
    description: str = ""
    bos_token: str = ""
    eos_token: str = ""
    unk_token: str = ""
    pad_token: str = ""
    supports_system: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "supports_system": self.supports_system,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatTemplate":
        """Create from dictionary."""
        return cls(**data)


# ============================================================================
# Built-in Templates
# ============================================================================

BUILTIN_TEMPLATES = {
    # ChatML format (used by many models including Qwen, DeepSeek, etc.)
    "chatml": ChatTemplate(
        name="chatml",
        template=(
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        ),
        description="ChatML format used by Qwen, DeepSeek, and others",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        supports_system=True,
    ),

    # Llama 2/3 format
    "llama": ChatTemplate(
        name="llama",
        template=(
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<s>[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' </s>' }}"
            "{% endif %}"
            "{% endfor %}"
        ),
        description="Llama 2/3 format with [INST] tags",
        bos_token="<s>",
        eos_token="</s>",
        supports_system=True,
    ),

    # Alpaca format
    "alpaca": ChatTemplate(
        name="alpaca",
        template=(
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "{{ '### Instruction:\n' + message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '### Response:\n' + message['content'] + '\n\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '### Response:\n' }}"
            "{% endif %}"
        ),
        description="Alpaca instruction format",
        supports_system=True,
    ),

    # Simple format for models without specific requirements
    "simple": ChatTemplate(
        name="simple",
        template=(
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: ' + message['content'] + '\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'Assistant: ' }}"
            "{% endif %}"
        ),
        description="Simple format with User:/Assistant: labels",
        supports_system=True,
    ),

    # TinyLlama specific format (similar to ChatML but simpler)
    "tinyllama": ChatTemplate(
        name="tinyllama",
        template=(
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '<|system|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>\n' + message['content'] + '\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\n' }}"
            "{% endif %}"
        ),
        description="TinyLlama format with role tags",
        supports_system=True,
    ),
}


# ============================================================================
# Model-to-Template Mapping
# ============================================================================

MODEL_TEMPLATE_MAPPING = {
    # Exact model names
    "TinyLlama/TinyLlama_v1.1": "tinyllama",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama",

    # Pattern-based matching (will be compiled to regex)
    r".*TinyLlama.*": "tinyllama",
    r".*Llama-2.*": "llama",
    r".*Llama-3.*": "llama",
    r".*Qwen.*": "chatml",
    r".*DeepSeek.*": "chatml",
    r".*Mistral.*": "llama",
    r".*Mixtral.*": "llama",
    r".*alpaca.*": "alpaca",
}


# ============================================================================
# Chat Template Manager
# ============================================================================

class ChatTemplateManager:
    """
    Manages chat templates for LLM tokenizers.

    This class provides:
    - Automatic template detection based on model name
    - Template registry for custom templates
    - Template application to tokenizers
    - Template validation
    - Persistence to JSON files

    Example:
        manager = ChatTemplateManager()
        manager.apply_template(tokenizer, "TinyLlama/TinyLlama_v1.1")
    """

    def __init__(
        self,
        custom_templates: Optional[Dict[str, ChatTemplate]] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
        template_file: Optional[Path] = None,
    ):
        """
        Initialize the template manager.

        Args:
            custom_templates: Additional custom templates to register
            custom_mapping: Additional model-to-template mappings
            template_file: Path to JSON file with templates (for persistence)
        """
        self.templates: Dict[str, ChatTemplate] = {}
        self.model_mapping: Dict[str, str] = {}

        # Load built-in templates
        for name, template in BUILTIN_TEMPLATES.items():
            self.templates[name] = template

        # Load built-in mappings
        for pattern, template_name in MODEL_TEMPLATE_MAPPING.items():
            self.model_mapping[pattern] = template_name

        # Add custom templates
        if custom_templates:
            for name, template in custom_templates.items():
                self.templates[name] = template

        # Add custom mappings
        if custom_mapping:
            self.model_mapping.update(custom_mapping)

        # Load from file if provided
        if template_file and template_file.exists():
            self.load_from_file(template_file)

        logger.info(f"ChatTemplateManager initialized with {len(self.templates)} templates")

    def register_template(
        self,
        name: str,
        template: Union[str, ChatTemplate],
        description: str = "",
        **kwargs
    ) -> None:
        """
        Register a custom template.

        Args:
            name: Template name
            template: Template string or ChatTemplate object
            description: Template description
            **kwargs: Additional template parameters (bos_token, eos_token, etc.)
        """
        if isinstance(template, str):
            template = ChatTemplate(
                name=name,
                template=template,
                description=description,
                **kwargs
            )

        self.templates[name] = template
        logger.info(f"Registered template: {name}")

    def register_model_mapping(self, model_pattern: str, template_name: str) -> None:
        """
        Register a model-to-template mapping.

        Args:
            model_pattern: Model name or regex pattern
            template_name: Template name to use
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Register it first.")

        self.model_mapping[model_pattern] = template_name
        logger.info(f"Registered mapping: {model_pattern} -> {template_name}")

    def get_template_for_model(self, model_name: str) -> Optional[ChatTemplate]:
        """
        Get the appropriate template for a model.

        Args:
            model_name: Model name or path

        Returns:
            ChatTemplate object or None if no match found
        """
        # Try exact match first
        if model_name in self.model_mapping:
            template_name = self.model_mapping[model_name]
            return self.templates.get(template_name)

        # Try pattern matching
        for pattern, template_name in self.model_mapping.items():
            if re.match(pattern, model_name):
                logger.debug(f"Matched {model_name} to pattern {pattern}")
                return self.templates.get(template_name)

        # Default fallback
        logger.warning(f"No template found for {model_name}, using 'simple' as fallback")
        return self.templates.get("simple")

    def apply_template(
        self,
        tokenizer: Any,
        model_name: Optional[str] = None,
        template_name: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Apply a chat template to a tokenizer.

        Args:
            tokenizer: Tokenizer object (from transformers)
            model_name: Model name (for automatic template selection)
            template_name: Explicit template name to use
            force: Force application even if tokenizer already has a template

        Returns:
            True if template was applied, False otherwise
        """
        # Check if tokenizer already has a template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template and not force:
            logger.info("Tokenizer already has a chat template, skipping")
            return False

        # Determine which template to use
        if template_name:
            # Explicit template name provided
            template = self.templates.get(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")
        elif model_name:
            # Auto-detect based on model name
            template = self.get_template_for_model(model_name)
            if not template:
                raise ValueError(f"No template found for model '{model_name}'")
        else:
            # Use simple fallback
            template = self.templates["simple"]
            logger.warning("No model_name or template_name provided, using 'simple' template")

        # Apply template
        try:
            tokenizer.chat_template = template.template

            # Apply special tokens if provided
            if template.bos_token and not tokenizer.bos_token:
                tokenizer.bos_token = template.bos_token
            if template.eos_token and not tokenizer.eos_token:
                tokenizer.eos_token = template.eos_token
            if template.pad_token and not tokenizer.pad_token:
                tokenizer.pad_token = template.pad_token
            if template.unk_token and not tokenizer.unk_token:
                tokenizer.unk_token = template.unk_token

            # Set padding side for decoder-only models
            if hasattr(tokenizer, 'padding_side'):
                tokenizer.padding_side = 'left'

            logger.info(f"Applied template '{template.name}' to tokenizer")
            return True

        except Exception as e:
            logger.error(f"Failed to apply template: {e}")
            return False

    def test_template(
        self,
        tokenizer: Any,
        template_name: str,
        test_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Test a template by applying it and formatting test messages.

        Args:
            tokenizer: Tokenizer object
            template_name: Template to test
            test_messages: Optional custom test messages

        Returns:
            Formatted output or None if failed
        """
        if test_messages is None:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "user", "content": "What can you help me with?"},
            ]

        template = self.templates.get(template_name)
        if not template:
            logger.error(f"Template '{template_name}' not found")
            return None

        # Temporarily apply template
        original_template = getattr(tokenizer, 'chat_template', None)

        try:
            self.apply_template(tokenizer, template_name=template_name, force=True)

            # Format messages
            formatted = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            logger.info(f"Template '{template_name}' test successful")
            return formatted

        except Exception as e:
            logger.error(f"Template test failed: {e}")
            return None

        finally:
            # Restore original template
            if original_template:
                tokenizer.chat_template = original_template

    def save_to_file(self, file_path: Path) -> None:
        """
        Save templates and mappings to a JSON file.

        Args:
            file_path: Path to save file
        """
        data = {
            "templates": {
                name: template.to_dict()
                for name, template in self.templates.items()
            },
            "mappings": self.model_mapping,
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved templates to {file_path}")

    def load_from_file(self, file_path: Path) -> None:
        """
        Load templates and mappings from a JSON file.

        Args:
            file_path: Path to load from
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Load templates
        for name, template_dict in data.get("templates", {}).items():
            self.templates[name] = ChatTemplate.from_dict(template_dict)

        # Load mappings
        self.model_mapping.update(data.get("mappings", {}))

        logger.info(f"Loaded templates from {file_path}")

    def list_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())

    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a template."""
        template = self.templates.get(template_name)
        if template:
            return template.to_dict()
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def auto_detect_and_apply_template(tokenizer: Any, model_name: str) -> bool:
    """
    Convenience function to auto-detect and apply template in one call.

    Args:
        tokenizer: Tokenizer object
        model_name: Model name

    Returns:
        True if template was applied
    """
    manager = ChatTemplateManager()
    return manager.apply_template(tokenizer, model_name=model_name)


def get_default_template(model_name: str) -> str:
    """
    Get the default template string for a model.

    Args:
        model_name: Model name

    Returns:
        Template string
    """
    manager = ChatTemplateManager()
    template = manager.get_template_for_model(model_name)
    return template.template if template else BUILTIN_TEMPLATES["simple"].template


# ============================================================================
# CLI Interface (for standalone usage)
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Chat Template Manager CLI")
    parser.add_argument("command", choices=["list", "info", "test"], help="Command to run")
    parser.add_argument("--template", help="Template name")
    parser.add_argument("--model", help="Model name")

    args = parser.parse_args()

    # Setup logging for CLI
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manager = ChatTemplateManager()

    if args.command == "list":
        print("Available templates:")
        for name in manager.list_templates():
            template = manager.templates[name]
            print(f"  - {name}: {template.description}")

    elif args.command == "info":
        if not args.template:
            print("Error: --template required for 'info' command")
            sys.exit(1)

        info = manager.get_template_info(args.template)
        if info:
            print(f"Template: {args.template}")
            print(f"Description: {info['description']}")
            print(f"Supports system: {info['supports_system']}")
            print(f"\nTemplate string:\n{info['template']}")
        else:
            print(f"Template '{args.template}' not found")

    elif args.command == "test":
        if not args.model:
            print("Error: --model required for 'test' command")
            sys.exit(1)

        template = manager.get_template_for_model(args.model)
        if template:
            print(f"Model '{args.model}' would use template: {template.name}")
            print(f"Description: {template.description}")
        else:
            print(f"No template found for model '{args.model}'")
