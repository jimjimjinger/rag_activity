"""네이티브 VL 모델을 생성하는 동안에만 체크포인트 텐서를 CPU에서 준비한다."""
from contextlib import contextmanager
from importlib.metadata import version
from math import prod


@contextmanager
def checkpoint_cpu_staging(paddle):
    # 이 두 로더 후킹 지점은 버전을 고정한 PaddleX 구현에 맞춰져 있다.
    if version("paddlex") != "3.7.2":
        raise RuntimeError("Review CPU checkpoint staging before changing PaddleX 3.7.2.")
    from paddlex.inference.models.common.transformers.transformers import model_utils

    original_load = model_utils.load_state_dict
    original_convert = model_utils._convert_state_dict_dtype_and_shape
    statistics = {"checkpoint_tensor_count": 0, "checkpoint_bytes": 0,
                  "conversion_on_cpu": False}

    def load_on_cpu(*args, **kwargs):
        with paddle.device_guard("cpu"):
            state = original_load(*args, **kwargs)
        tensors = [value for value in state.values() if isinstance(value, paddle.Tensor)]
        if not tensors or not all(value.place.is_cpu_place() for value in tensors):
            raise RuntimeError("Checkpoint tensors were not staged on CPU.")
        statistics["checkpoint_tensor_count"] += len(tensors)
        statistics["checkpoint_bytes"] += sum(int(prod(value.shape) * value.element_size()) for value in tensors)
        return state

    def convert_on_cpu(state, model, convert_from_hf):
        with paddle.device_guard("cpu"):
            result = original_convert(state, model, convert_from_hf)
        if not all(value.place.is_cpu_place() for value in state.values() if isinstance(value, paddle.Tensor)):
            raise RuntimeError("Converted checkpoint tensors left the CPU staging area.")
        statistics["conversion_on_cpu"] = True
        return result

    model_utils.load_state_dict = load_on_cpu
    model_utils._convert_state_dict_dtype_and_shape = convert_on_cpu
    try:
        yield statistics
    finally:
        model_utils.load_state_dict = original_load
        model_utils._convert_state_dict_dtype_and_shape = original_convert
