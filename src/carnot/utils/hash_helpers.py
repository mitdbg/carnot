import hashlib
import json


def hash_for_id(id_str: str, max_chars: int = 16) -> str:
    return hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:max_chars]


def hash_for_serialized_dict(dict_obj: dict) -> str:
    return hash_for_id(json.dumps(dict_obj, sort_keys=True))
