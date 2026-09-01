"""
Exports both encoders to ONNX so serving needs no deep learning framework.

Pooling and normalisation are baked into the graph rather than reproduced at
call time: the two have to agree exactly with what the model was trained under,
and a graph that carries them cannot drift from a reimplementation.

Usage: python -m tools.export_encoders
"""

import json
import shutil
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch.export import Dim
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from ingest import config
from retrieval import config as retrieval_config

OUT = config.ROOT / "artifacts" / "onnx"


class Passages(torch.nn.Module):
    """CLS pooling and L2 normalisation, which is what the bi-encoder was trained with."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = hidden[:, 0]
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)


class Pairs(torch.nn.Module):
    """One relevance logit per query and passage read together."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits


def strip_stale_shapes(path: Path) -> None:
    """
    Drops the shape metadata the exporter leaves behind.

    It records shapes for intermediate tensors and then optimises the graph
    without updating them, which later readers reject as contradictory.
    """
    graph = onnx.load(str(path))
    del graph.graph.value_info[:]
    onnx.save(graph, str(path))


def export(module, sample, names: list[str], target: Path) -> None:
    axes = {0: Dim("batch"), 1: Dim("sequence")}
    torch.onnx.export(
        module,
        tuple(sample[name] for name in names),
        str(target),
        input_names=names,
        output_names=["output"],
        dynamic_shapes=tuple(axes for _ in names),
        external_data=False,
    )
    strip_stale_shapes(target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for name, folder, builder, wrapper, inputs in (
        (
            config.EMBEDDING_MODEL,
            "passages",
            AutoModel.from_pretrained,
            Passages,
            ["input_ids", "attention_mask"],
        ),
        (
            retrieval_config.RERANKER_MODEL,
            "pairs",
            AutoModelForSequenceClassification.from_pretrained,
            Pairs,
            ["input_ids", "attention_mask", "token_type_ids"],
        ),
    ):
        target = OUT / folder
        target.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = builder(name).eval()
        sample = tokenizer(
            ["a query", "another query"],
            ["a passage about key rates", "a passage about satellites"],
            padding=True,
            return_tensors="pt",
        )
        if "token_type_ids" not in sample:
            sample["token_type_ids"] = torch.zeros_like(sample["input_ids"])

        export(wrapper(model), sample, inputs, target / "model-fp32.onnx")

        # Dynamic quantisation touches the weights only; activation ranges are
        # computed per call, so no calibration set is needed.
        quantize_dynamic(
            str(target / "model-fp32.onnx"),
            str(target / "model.onnx"),
            weight_type=QuantType.QInt8,
        )

        source = Path(tokenizer.name_or_path)
        tokenizer.save_pretrained(str(target))
        for extra in ("tokenizer.json",):
            candidate = source / extra
            if candidate.exists():
                shutil.copy(candidate, target / extra)

        (target / "export.json").write_text(
            json.dumps({"model": name, "inputs": inputs}, indent=2) + "\n"
        )
        full = (target / "model-fp32.onnx").stat().st_size / 1e6
        small = (target / "model.onnx").stat().st_size / 1e6
        print(f"{folder:<12}{name:<44}{full:8.1f} MB fp32 {small:8.1f} MB int8")


if __name__ == "__main__":
    main()
