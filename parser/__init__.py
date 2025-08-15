"""
Parser package for spectral detection results
"""

from .base_parser import BaseParser
from .parser_manager import ParserManager
from .spectral_parser import SpectralParser

__all__ = [
    'BaseParser', 
    'ParserManager', 
    'SpectralParser',
] 