# src/models/sliding_window_attention.py
"""
Sliding Window Attention Module

Implements dynamic context management with sliding windows for long sequences.
Reduces memory usage by maintaining only a fixed-size window of recent tokens
while optionally preserving important context (e.g., system prompts).

This module can be used to monkey-patch transformer models to support
longer sequences with reduced memory footprint.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window attention."""
    window_size: int = 2048  # Size of the sliding window
    stride: int = 512  # How many tokens to slide when window is full
    keep_prefix_tokens: int = 128  # Number of prefix tokens to always keep (e.g., system prompt)
    enable_compression: bool = False  # Enable context compression (future feature)
    compression_ratio: float = 0.5  # Compression ratio for old context
    device: str = "cuda"  # Device for attention computation


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention mechanism that maintains a fixed-size context window.

    This module wraps standard attention and manages the context window,
    automatically sliding when the window is full and optionally preserving
    important prefix tokens.
    """

    def __init__(self, config: SlidingWindowConfig):
        """
        Initialize sliding window attention.

        Args:
            config: Configuration for sliding window
        """
        super().__init__()
        self.config = config

        # Track current window state
        self.current_position = 0
        self.total_tokens_seen = 0

        # Statistics
        self.num_slides = 0
        self.tokens_discarded = 0
        self.tokens_preserved = 0

        logger.info(f"SlidingWindowAttention initialized: window_size={config.window_size}, "
                   f"stride={config.stride}, keep_prefix={config.keep_prefix_tokens}")

    def apply_sliding_window(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply sliding window to key-value states.

        Args:
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            Tuple of (windowed_keys, windowed_values, windowed_mask, windowed_position_ids)
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Update total tokens
        self.total_tokens_seen += seq_len

        # If sequence is within window size, no sliding needed
        if seq_len <= self.config.window_size:
            self.current_position = seq_len
            return key_states, value_states, attention_mask, position_ids

        # Need to apply sliding window
        logger.debug(f"Applying sliding window: seq_len={seq_len}, window_size={self.config.window_size}")

        # Calculate window boundaries
        if self.config.keep_prefix_tokens > 0:
            # Keep prefix + sliding window of recent tokens
            prefix_size = min(self.config.keep_prefix_tokens, seq_len)
            remaining_window = self.config.window_size - prefix_size

            if seq_len <= prefix_size + remaining_window:
                # Sequence fits in prefix + window
                windowed_keys = key_states
                windowed_values = value_states
                windowed_mask = attention_mask
                windowed_position_ids = position_ids
            else:
                # Need to slide: keep prefix + recent tokens
                # Take prefix tokens
                prefix_keys = key_states[:, :, :prefix_size, :]
                prefix_values = value_states[:, :, :prefix_size, :]

                # Take most recent tokens
                recent_start = seq_len - remaining_window
                recent_keys = key_states[:, :, recent_start:, :]
                recent_values = value_states[:, :, recent_start:, :]

                # Concatenate prefix + recent
                windowed_keys = torch.cat([prefix_keys, recent_keys], dim=2)
                windowed_values = torch.cat([prefix_values, recent_values], dim=2)

                # Update mask if provided
                if attention_mask is not None:
                    prefix_mask = attention_mask[:, :, :prefix_size, :]
                    recent_mask = attention_mask[:, :, recent_start:, :]
                    windowed_mask = torch.cat([prefix_mask, recent_mask], dim=2)
                else:
                    windowed_mask = None

                # Update position IDs if provided
                if position_ids is not None:
                    prefix_pos = position_ids[:, :prefix_size]
                    recent_pos = position_ids[:, recent_start:]
                    windowed_position_ids = torch.cat([prefix_pos, recent_pos], dim=1)
                else:
                    windowed_position_ids = None

                # Track statistics
                self.num_slides += 1
                self.tokens_discarded += (recent_start - prefix_size)
                self.tokens_preserved += prefix_size

                logger.debug(f"Slid window: discarded {recent_start - prefix_size} tokens, "
                           f"kept {prefix_size} prefix + {remaining_window} recent")
        else:
            # No prefix preservation - simple sliding window
            if seq_len <= self.config.window_size:
                windowed_keys = key_states
                windowed_values = value_states
                windowed_mask = attention_mask
                windowed_position_ids = position_ids
            else:
                # Take most recent tokens only
                start_pos = seq_len - self.config.window_size
                windowed_keys = key_states[:, :, start_pos:, :]
                windowed_values = value_states[:, :, start_pos:, :]

                if attention_mask is not None:
                    windowed_mask = attention_mask[:, :, start_pos:, :]
                else:
                    windowed_mask = None

                if position_ids is not None:
                    windowed_position_ids = position_ids[:, start_pos:]
                else:
                    windowed_position_ids = None

                # Track statistics
                self.num_slides += 1
                self.tokens_discarded += start_pos

                logger.debug(f"Slid window: discarded {start_pos} tokens, kept {self.config.window_size} recent")

        # Update current position
        self.current_position = windowed_keys.shape[2]

        return windowed_keys, windowed_values, windowed_mask, windowed_position_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        original_attention_fn: Optional[callable] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.

        This method is designed to wrap an existing attention module.

        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Past key-value cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            original_attention_fn: Original attention forward function to call
            **kwargs: Additional arguments

        Returns:
            Tuple of (attention_output, past_key_value, attention_weights)
        """
        if original_attention_fn is None:
            raise ValueError("original_attention_fn must be provided for sliding window attention")

        # Call original attention to get key-value states
        # This is a simplified version - actual implementation would need to extract K,V
        # from the attention module before computing attention

        # For now, pass through to original attention
        # Full implementation would require deeper integration with specific attention implementations
        outputs = original_attention_fn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )

        # Extract outputs
        if isinstance(outputs, tuple):
            attention_output = outputs[0]
            new_past_key_value = outputs[1] if len(outputs) > 1 else None
            attention_weights = outputs[2] if len(outputs) > 2 else None
        else:
            attention_output = outputs
            new_past_key_value = None
            attention_weights = None

        # Apply sliding window to past_key_value if present
        if new_past_key_value is not None and use_cache:
            key_states, value_states = new_past_key_value
            windowed_keys, windowed_values, _, _ = self.apply_sliding_window(
                key_states, value_states, attention_mask, position_ids
            )
            new_past_key_value = (windowed_keys, windowed_values)

        return attention_output, new_past_key_value, attention_weights

    def reset(self):
        """Reset window state."""
        self.current_position = 0
        self.total_tokens_seen = 0
        self.num_slides = 0
        self.tokens_discarded = 0
        self.tokens_preserved = 0
        logger.debug("Sliding window state reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about sliding window usage."""
        return {
            'window_size': self.config.window_size,
            'current_position': self.current_position,
            'total_tokens_seen': self.total_tokens_seen,
            'num_slides': self.num_slides,
            'tokens_discarded': self.tokens_discarded,
            'tokens_preserved': self.tokens_preserved,
            'effective_compression_ratio': (
                self.tokens_discarded / self.total_tokens_seen
                if self.total_tokens_seen > 0 else 0.0
            )
        }

    def format_statistics(self) -> str:
        """Format statistics for display."""
        stats = self.get_statistics()

        lines = [
            "",
            "━" * 60,
            "🪟 Sliding Window Attention Statistics",
            "━" * 60,
            f"Window Size: {stats['window_size']} tokens",
            f"Current Position: {stats['current_position']} tokens",
            f"Total Tokens Seen: {stats['total_tokens_seen']} tokens",
            "",
            f"Number of Slides: {stats['num_slides']}",
            f"Tokens Discarded: {stats['tokens_discarded']}",
            f"Tokens Preserved: {stats['tokens_preserved']}",
            f"Effective Compression: {stats['effective_compression_ratio'] * 100:.1f}%",
            "━" * 60,
            ""
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (f"SlidingWindowAttention(window_size={self.config.window_size}, "
                f"position={self.current_position}, slides={self.num_slides})")
