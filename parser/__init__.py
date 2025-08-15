"""
Parser package for spectral detection results
"""

from .base_parser import BaseParser
from .spectralParser import SpectralParser
from .defaultParser import DefaultParser
__all__ = [
    'BaseParser', 
    'ParserManager', 
    'SpectralParser',
    'defaultParser'
] 