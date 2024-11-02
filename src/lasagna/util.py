from urllib.parse import urlparse
import mimetypes
import asyncio
import hashlib
import base64
import aiohttp
import os
import re

from typing import Tuple, List, Literal, Union, TypeVar, Callable, Protocol, Any
from typing_extensions import Buffer

from .types import ToolParam, ImageMimeTypes


T = TypeVar('T')


def parse_docstring(docstring: str) -> Tuple[str, List[ToolParam]]:
    lines = docstring.splitlines()
    indent_amounts = [(len(line) - len(line.lstrip())) for line in lines]
    first_indent = max(ia for ia in indent_amounts[:2])
    lines_stripped = []
    for ia, l in zip(indent_amounts, lines):
        to_strip = min(ia, first_indent, len(l))
        lines_stripped.append(l[to_strip:].rstrip())
    dedented_docs = "\n".join(lines_stripped).strip()
    if ':param:' not in dedented_docs:
        # Special case where there are no parameters!
        description = ' '.join(dedented_docs.strip().splitlines())
        return description, []
    description_match = re.search(r"^(.*?)(?=:param:)", dedented_docs, re.DOTALL)
    if description_match is None:
        raise ValueError("no description found")
    description = ' '.join(description_match[1].strip().splitlines())
    if not description:
        raise ValueError("no description found")
    params_found = re.findall(r":param:\s+(\w+):\s+([\w ]+):\s+(.+)", dedented_docs)
    params: List[ToolParam] = [
        {
            'name': p[0].strip(),
            'type': p[1].strip(),
            'description': p[2].strip(),
        }
        for p in params_found
    ]
    for p in params:
        if not p['name']:
            raise ValueError("no parameter name found")
        if not p['type'].startswith('enum ') and p['type'] not in ['str', 'float', 'int', 'bool']:
            raise ValueError(f"invalid type found: {p['type']}")
        if not p['description']:
            del p['description']
        else:
            if p['description'].startswith('(optional)'):
                p['optional'] = True
    return description, params


def combine_pairs(
    lst: List[T],
    decision_func: Callable[[T, T], Union[Literal[False], Tuple[Literal[True], T]]],
) -> List[T]:
    # This implementation is ugly but I can't find a nice way to do it.
    res: List[T] = []
    i: int = 0
    did_just_combine: bool = False
    while i < len(lst) - 1:
        j = i + 1
        flag = decision_func(lst[i], lst[j])
        if flag is False:
            if not did_just_combine:
                res.append(lst[i])
            did_just_combine = False
        else:
            new_val = flag[1]
            res.append(new_val)
            did_just_combine = True
        i += 1
    if not did_just_combine and i < len(lst):
        # If the last pair was *not* combined, then we need to add the last element.
        res.append(lst[i])
    return res


def _is_remote_or_local(image_filepath_or_url: str) -> Tuple[bool, str]:
    parsed_url = urlparse(image_filepath_or_url)
    if parsed_url.scheme in ('http', 'https'):
        # The string is already a remote URL, so just return it.
        return True, image_filepath_or_url

    if image_filepath_or_url.startswith('file://'):
        local_path = image_filepath_or_url[7:]
    else:
        local_path = image_filepath_or_url

    if not os.path.isfile(local_path):
        raise ValueError(f'cannot find file: {image_filepath_or_url}')

    return False, local_path


def _read_as_base64(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def _http_get_as_base64(url: str) -> str:
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url) as response:
            data = await response.read()
            return base64.b64encode(data).decode('utf-8')


def _get_image_mimetype(path: str) -> ImageMimeTypes:
    mimetype = mimetypes.guess_type(path)[0]
    if mimetype is None:
        raise ValueError(f"unknown mimetype for file: {path}")
    return mimetype  # type: ignore


def convert_to_image_url_sync(image_filepath_or_url: str) -> str:
    is_remote, image_filepath_or_url = _is_remote_or_local(image_filepath_or_url)
    if is_remote:
        return image_filepath_or_url
    else:
        local_path = image_filepath_or_url

    mimetype = _get_image_mimetype(local_path)
    img_encoded = _read_as_base64(local_path)

    return f"data:{mimetype};base64,{img_encoded}"


async def convert_to_image_url(image_filepath_or_url: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, convert_to_image_url_sync, image_filepath_or_url)


async def convert_to_image_base64(image_filepath_or_url: str) -> Tuple[ImageMimeTypes, str]:
    is_remote, image_filepath_or_url = _is_remote_or_local(image_filepath_or_url)
    mimetype = _get_image_mimetype(image_filepath_or_url)
    if is_remote:
        data = await _http_get_as_base64(image_filepath_or_url)
    else:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _read_as_base64, image_filepath_or_url)
    return mimetype, data


def exponential_backoff_retry_delays(
    n_total_tries: int,
    base: float = 2.0,
    max_delay: float = 60.0,
) -> List[float]:
    assert n_total_tries > 0
    delay_list = [(base ** exponent) for exponent in range(1, n_total_tries + 1)]
    delay_list[-1] = 0.0    # <-- after the *last* failure, you should not delay at all
    return [min(d, max_delay) for d in delay_list]


def get_name(obj: Any) -> str:
    name = str(obj.__name__) if hasattr(obj, '__name__') else str(obj)
    return name


class HashAlgorithm(Protocol):
    """
    All the hashlib algorithms conform to this, like:
      - hashlib.md5
      - hashlib.sha1
      - hashlib.sha256
      - ...
    """
    def update(self, data: Buffer, /) -> None: ...
    def hexdigest(self) -> str: ...


HashAlgorithmFactory = Callable[[], HashAlgorithm]


def recursive_hash(
    seed: Union[str, None],
    obj: Any,
    alg_factory: HashAlgorithmFactory = lambda: hashlib.sha256(),
) -> str:
    alg = alg_factory()
    def _r(obj: Any) -> None:
        if isinstance(obj, dict):
            items = sorted(obj.items())  # <-- it's critical to sort them so we get canonical hashes
            alg.update('__open_dict__'.encode('utf-8'))
            _r(items)
            alg.update('__close_dict__'.encode('utf-8'))
        elif isinstance(obj, (list, tuple)):
            alg.update('__open_list__'.encode('utf-8'))
            for o in obj:
                _r(o)
            alg.update('__close_list__'.encode('utf-8'))
        elif isinstance(obj, str):
            alg.update('__str__'.encode('utf-8'))
            alg.update(obj.encode('utf-8'))
        elif isinstance(obj, bool):
            alg.update('__bool__'.encode('utf-8'))
            alg.update(f'{1 if obj else 0}'.encode('utf-8'))
        elif isinstance(obj, int):
            alg.update('__int__'.encode('utf-8'))
            alg.update(f'{obj}'.encode('utf-8'))
        elif isinstance(obj, float):
            alg.update('__float__'.encode('utf-8'))
            alg.update(f'{obj:.8e}'.encode('utf-8'))
        elif obj is None:
            alg.update('__None__'.encode('utf-8'))
        else:
            raise ValueError(f"unsupported type: {type(obj)}")
    if seed is not None:
        _r(seed)
    _r(obj)
    return alg.hexdigest()
