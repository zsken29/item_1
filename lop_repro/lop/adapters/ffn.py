from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FfnLayerSpec:
    """单层 FFN 的规格描述。

    Attributes:
        index: 层在模型中的序号。
        block_path: 该层权重在状态字典中的前缀路径，如 ``model.layers.0``。
        activation_path: 激活值对应的模块路径，用于定位中间特征。
        gate_proj_weight: gate_proj 权重键名。
        up_proj_weight: up_proj 权重键名。
        down_proj_weight: down_proj 权重键名。
        hidden_size: 隐藏层维度。
        intermediate_size: 中间层维度。
        weight_status: 权重索引验证结果，如 ``verified``、``missing:...`` 或 ``not_indexed``。
    """

    index: int
    block_path: str
    activation_path: str
    gate_proj_weight: str
    up_proj_weight: str
    down_proj_weight: str
    hidden_size: int
    intermediate_size: int
    weight_status: str


@dataclass(frozen=True)
class ModelFfnSpec:
    """模型 FFN 结构的完整规格。

    Attributes:
        name: 模型目录名称。
        model_type: 配置中的 model_type，如 ``qwen2_5_vl``。
        architecture: 配置中的 architectures 值。
        supported: 当前是否支持对该模型进行 FFN 剪枝。
        layers: 所有 FFN 层的规格元组。
        note: 对该模型 FFN 结构的补充说明。
    """

    name: str
    model_type: str
    architecture: str
    supported: bool
    layers: tuple[FfnLayerSpec, ...]
    note: str


def inspect_model_ffn(model_dir: str | Path) -> ModelFfnSpec:
    """从模型目录解析 FFN 结构规格。

    读取 ``config.json`` 和 ``model.safetensors.index.json``，
    根据 ``model_type`` 分派到对应的解析逻辑。

    Args:
        model_dir: 模型目录路径。

    Returns:
        该模型的 FFN 完整规格。

    Raises:
        FileNotFoundError: 缺少 ``config.json`` 时抛出。
        ValueError: 遇到不支持的 ``model_type`` 时抛出。
    """
    path = Path(model_dir)
    config = _read_json(path / "config.json")
    model_type = config["model_type"]
    weight_keys = _read_weight_index(path)

    if model_type == "qwen2_5_vl":
        return _dense_qwen_spec(
            name=path.name,
            model_type=model_type,
            architecture=config["architectures"][0],
            layer_count=int(config["num_hidden_layers"]),
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            block_prefix="model.language_model.layers",
            weight_prefix="model.layers",
            weight_keys=weight_keys,
            supported=True,
            note="Qwen2.5-VL language FFN uses dense gate/up/down projections.",
        )

    if model_type == "internvl_chat":
        llm_config = config["llm_config"]
        if llm_config["architectures"][0] != "Qwen2ForCausalLM":
            raise ValueError(f"{path}: expected Qwen2ForCausalLM llm_config")
        return _dense_qwen_spec(
            name=path.name,
            model_type=model_type,
            architecture=config["architectures"][0],
            layer_count=int(llm_config["num_hidden_layers"]),
            hidden_size=int(llm_config["hidden_size"]),
            intermediate_size=int(llm_config["intermediate_size"]),
            block_prefix="language_model.model.layers",
            weight_keys=weight_keys,
            supported=True,
            note="InternVL chat wraps a Qwen2 language model under language_model.",
        )

    if model_type == "deepseek_vl_v2":
        language_config = config["language_config"]
        return _deepseek_vl2_spec(path.name, model_type, config["model_type"], language_config, weight_keys)

    raise ValueError(f"{path}: unsupported model_type {model_type!r}")


def _dense_qwen_spec(
    name: str,
    model_type: str,
    architecture: str,
    layer_count: int,
    hidden_size: int,
    intermediate_size: int,
    block_prefix: str,
    weight_keys: set[str],
    supported: bool,
    note: str,
    weight_prefix: str | None = None,
) -> ModelFfnSpec:
    """为使用 dense FFN 的 Qwen 系列模型构建规格。"""
    if weight_prefix is None:
        weight_prefix = block_prefix
    layers = tuple(
        _dense_layer(index, block_prefix, weight_prefix, hidden_size, intermediate_size, weight_keys)
        for index in range(layer_count)
    )
    return ModelFfnSpec(
        name=name,
        model_type=model_type,
        architecture=architecture,
        supported=supported,
        layers=layers,
        note=note,
    )


def _deepseek_vl2_spec(
    name: str,
    model_type: str,
    architecture: str,
    language_config: dict,
    weight_keys: set[str],
) -> ModelFfnSpec:
    """为 DeepSeek-VL2 构建规格。

    该模型在第 0 层之后切换为 MoE FFN，dense FFN 剪枝暂未实现。
    """
    first = _dense_layer(
        index=0,
        block_prefix="language.model.layers",
        weight_prefix="language.model.layers",
        hidden_size=int(language_config["hidden_size"]),
        intermediate_size=int(language_config["intermediate_size"]),
        weight_keys=weight_keys,
    )
    return ModelFfnSpec(
        name=name,
        model_type=model_type,
        architecture=architecture,
        supported=False,
        layers=(first,),
        note="DeepSeek-VL2-Tiny switches to MoE FFN after layer 0; dense FFN pruning is not implemented for it.",
    )


def _dense_layer(
    index: int,
    block_prefix: str,
    weight_prefix: str,
    hidden_size: int,
    intermediate_size: int,
    weight_keys: set[str],
) -> FfnLayerSpec:
    """根据层序号和前缀构造单层 dense FFN 规格。

    block_prefix 用于模块访问路径 (get_submodule, activation hook)，
    weight_prefix 用于权重索引键的验证 (safetensors index)。
    """
    block_path = f"{block_prefix}.{index}"
    weight_block_path = f"{weight_prefix}.{index}"
    gate = f"{weight_block_path}.mlp.gate_proj.weight"
    up = f"{weight_block_path}.mlp.up_proj.weight"
    down = f"{weight_block_path}.mlp.down_proj.weight"
    return FfnLayerSpec(
        index=index,
        block_path=block_path,
        activation_path=f"{block_path}.mlp.down_proj",
        gate_proj_weight=gate,
        up_proj_weight=up,
        down_proj_weight=down,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        weight_status=_weight_status(weight_keys, (gate, up, down)),
    )


def _weight_status(weight_keys: set[str], required: tuple[str, str, str]) -> str:
    """检查所需权重键在索引中的存在状态。

    Args:
        weight_keys: 从 ``model.safetensors.index.json`` 读取的键集合。
        required: 需要验证的三个权重键名。

    Returns:
        ``verified`` 表示全部存在；``missing:...`` 列出缺失键；
        ``not_indexed`` 表示未找到索引文件。
    """
    if not weight_keys:
        return "not_indexed"
    missing = [key for key in required if key not in weight_keys]
    if missing:
        return "missing:" + ",".join(missing)
    return "verified"


def _read_json(path: Path) -> dict:
    """读取 JSON 文件并返回字典。"""
    if not path.is_file():
        raise FileNotFoundError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_weight_index(model_dir: Path) -> set[str]:
    """从 ``model.safetensors.index.json`` 中读取所有权重键名。

    若索引文件不存在，返回空集合。
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return set()
    index = _read_json(index_path)
    return set(index["weight_map"])
