from typing import List, Dict, Any

def _prepare_samples(*args) -> List[Dict[str, Any]]:
        """
        将各种输入格式统一处理为 dict 列表
        支持以下三种情况：
        1. list image_paths, str user_prompt, str system_prompt
        2. list image_paths, list user_prompts, list system_prompts
        3. list of dict [{"image_path":..., "user_prompt":..., "system_prompt":...}, ...]
        """
        if len(args) == 3:
            first, second, third = args
            if isinstance(first, list) and isinstance(second, str) and isinstance(third, str):
                # case 1: list, str, str
                return [{"image_path": p, "user_prompt": second, "system_prompt": third} for p in first]
            elif isinstance(first, list) and isinstance(second, list) and isinstance(third, list):
                # case 2: list, list, list
                if not (len(first) == len(second) == len(third)):
                    raise ValueError("Lengths of image_paths, user_prompts, system_prompts must match")
                return [{"image_path": p, "user_prompt": u, "system_prompt": s} for p, u, s in zip(first, second, third)]
            else:
                raise TypeError("Unsupported argument types for 3 args")
        elif len(args) == 1:
            first = args[0]
            if isinstance(first, list) and all(isinstance(x, dict) for x in first):
                # case 3: list of dict
                return [{"image_path": x.get("image_path"),
                         "user_prompt": x.get("user_prompt"),
                         "system_prompt": x.get("system_prompt")} for x in first]
            else:
                raise TypeError("Unsupported single argument type")
        else:
            raise TypeError("Unsupported number of arguments")