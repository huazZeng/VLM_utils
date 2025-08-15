from base_parser import BaseParser

class DefaultParser(BaseParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_to_print(self, raw_result: str, **kwargs) -> str:
        return raw_result
    
    def parse_to_save(self, raw_result: str, **kwargs) -> Dict[str, Any]:
        return raw_result