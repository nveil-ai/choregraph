# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Security utilities for path validation and sanitization."""
import os
from pathlib import Path
from typing import Union


def safe_path(path_or_base: Union[str, Path], user_input: Union[str, Path, None] = None) -> Path:
    """
    Validate and sanitize a path to prevent directory traversal attacks.
    
    Can be called in two modes:
    
    1. **Single-arg** ``safe_path(path)`` — resolves and normalizes the path,
       removing any ``..`` components.  Use this when the path is already
       constructed from trusted + dynamic parts and you just need to
       canonicalize it.
    
    2. **Two-arg** ``safe_path(base_dir, user_input)`` — resolves *user_input*
       relative to *base_dir* and verifies the result stays within
       *base_dir*.
    
    Args:
        path_or_base: A path to normalize (single-arg), or the base directory
            to validate against (two-arg).
        user_input: The user-provided path (relative or absolute).
            When ``None``, the function operates in single-arg mode.
    
    Returns:
        A safe, fully resolved :class:`~pathlib.Path`.
    
    Raises:
        ValueError: If the resulting path is outside the base directory
            (two-arg mode only).
    """
    if user_input is None:
        # Single-arg mode: normalize & resolve
        return Path(path_or_base).resolve()

    # Two-arg mode: validate containment
    base_dir = Path(path_or_base).resolve()
    user_path = Path(user_input)

    if user_path.is_absolute():
        full_path = user_path.resolve()
    else:
        full_path = (base_dir / user_path).resolve()

    try:
        full_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"Path {user_input} is outside the allowed directory {base_dir}")

    return full_path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to remove potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize.
    
    Returns:
        A sanitized filename.
    """
    # Remove any path traversal sequences
    sanitized = os.path.basename(filename)
    
    # Remove any control characters or other dangerous characters
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in ('.', '-', '_'))
    
    return sanitized
