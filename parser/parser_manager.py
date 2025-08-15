#!/usr/bin/env python3
"""
Parser Manager classes for coordinating multiple parsers.
"""

from typing import Dict, Any, List, Optional, Union
from .base_parser import BaseParser
from .parser_factory import ParserFactory


class ParserManager:
    """Manager class for coordinating multiple parsers."""
    
    def __init__(self, parsers: Optional[List[Union[str, BaseParser]]] = None, 
                 parser_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize parser manager.
        
        Args:
            parsers: List of parser types or parser instances
            parser_configs: Configuration for each parser type
        """
        self.parsers = []
        self.parser_configs = parser_configs or {}
        
        if parsers:
            for parser in parsers:
                if isinstance(parser, str):
                    # Create parser from type string
                    config = self.parser_configs.get(parser, {})
                    parser_instance = ParserFactory.create_parser(parser, **config)
                    self.parsers.append(parser_instance)
                elif isinstance(parser, BaseParser):
                    # Use provided parser instance
                    self.parsers.append(parser)
                else:
                    raise ValueError(f"Invalid parser: {parser}")
    
    def add_parser(self, parser_type: str, **kwargs) -> None:
        """
        Add a new parser.
        
        Args:
            parser_type: Type of parser to add
            **kwargs: Configuration parameters for the parser
        """
        parser_instance = ParserFactory.create_parser(parser_type, **kwargs)
        self.parsers.append(parser_instance)
    
    def parse_to_print(self, raw_result: str, **kwargs) -> List[str]:
        """
        Parse raw result to print format using all parsers.
        
        Args:
            raw_result: Raw result string to parse
            **kwargs: Additional parsing parameters
            
        Returns:
            List of formatted print strings from all parsers
        """
        results = []
        for parser in self.parsers:
            if parser.validate_result(raw_result):
                result = parser.parse_to_print(raw_result, **kwargs)
                results.append(result)
        return results
    
    def parse_to_save(self, raw_result: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Parse raw result to save format using all parsers.
        
        Args:
            raw_result: Raw result string to parse
            **kwargs: Additional parsing parameters
            
        Returns:
            List of structured data dictionaries from all parsers
        """
        results = []
        for parser in self.parsers:
            if parser.validate_result(raw_result):
                result = parser.parse_to_save(raw_result, **kwargs)
                results.append(result)
        return results
    
    def get_parser_names(self) -> List[str]:
        """
        Get names of all parsers.
        
        Returns:
            List of parser names
        """
        return [parser.get_parser_name() for parser in self.parsers]
    
    def get_parser_by_name(self, parser_name: str) -> Optional[BaseParser]:
        """
        Get parser instance by name.
        
        Args:
            parser_name: Name of the parser
            
        Returns:
            Parser instance or None if not found
        """
        for parser in self.parsers:
            if parser.get_parser_name() == parser_name:
                return parser
        return None
    
    def remove_parser(self, parser_name: str) -> bool:
        """
        Remove parser by name.
        
        Args:
            parser_name: Name of the parser to remove
            
        Returns:
            True if parser was removed, False if not found
        """
        for i, parser in enumerate(self.parsers):
            if parser.get_parser_name() == parser_name:
                del self.parsers[i]
                return True
        return False
    
    def clear_parsers(self) -> None:
        """Remove all parsers."""
        self.parsers.clear()
    
    @staticmethod
    def get_available_parsers() -> List[str]:
        """Get list of available parser types."""
        return ParserFactory.list_available_parsers()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current parser manager status.
        
        Returns:
            Dictionary with parser manager status
        """
        return {
            "parser_count": len(self.parsers),
            "parser_names": self.get_parser_names(),
            "available_parsers": self.get_available_parsers()
        } 