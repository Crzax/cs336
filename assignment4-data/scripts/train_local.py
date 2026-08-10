"""本地（非 modal）多卡训练启动脚本。

背景：官方 scripts/train.py 顶层 `from cs336_data.modal_utils import app,
build_image, VOLUME_MOUNTS` 以及 cs336_data.common 都会 `import modal`。
在本机（无 modal 库 / SUNET_ID 未设）裸跑 torchrun 时会在 import 阶段崩溃。

本脚本在 import scripts.train 之前，先把 modal 相关模块替换成兼容的 dummy，
从而让官方训练逻辑 train_from_config 能在本地跑通。不改动任何官方文件，
训练逻辑、模型结构、优化器、数据切分完全复用 cs336_basics。

用法（单卡）:
  uv run python scripts/train_local.py \
      --train-bin /path/to/data.bin \
      --valid-bin /path/to/validation.bin \
      --model-output /path/to/output

用法（8 卡 DDP）:
  torchrun --standalone --nproc_per_node=8 scripts/train_local.py \
      --train-bin /path/to/data.bin \
      --valid-bin /path/to/validation.bin \
      --model-output /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# 在 import scripts.train 之前，屏蔽 modal 依赖。
# 只有当系统里真的有 modal 库且能正常使用时才保留真实 modal；
# 否则注入兼容 dummy，保证本地 torchrun 能 import 通过。
# ---------------------------------------------------------------------------
def _install_modal_fallback() -> None:
    try:
        import modal  # noqa: F401
        _HAS_MODAL = True
    except Exception:  # noqa: BLE001
        _HAS_MODAL = False

    if _HAS_MODAL:
        return

    # dummy app：提供 function() 和 local_entrypoint() 两个装饰器方法
    class _DummyApp:
        def function(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

        def local_entrypoint(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

    dummy_modal = types.ModuleType("modal")
    dummy_modal.App = lambda *a, **k: None
    dummy_modal.Volume = type("Volume", (), {"from_name": classmethod(lambda *a, **k: None), "read_only": lambda s: s})
    dummy_modal.Image = type("Image", (), {})
    dummy_modal.CloudBucketMount = object

    sys.modules.setdefault("modal", dummy_modal)

    # cs336_data.common 里 import 了 modal（只用 modal.is_local()），给个足够兼容的版本
    dummy_app = _DummyApp()
    dummy_modal_utils = types.ModuleType("cs336_data.modal_utils")
    dummy_modal_utils.SUNET_ID = "TODO"
    dummy_modal_utils.app = dummy_app
    dummy_modal_utils.build_image = lambda *a, **k: None
    dummy_modal_utils.VOLUME_MOUNTS = {}
    dummy_modal_utils.MODAL_SECRETS = []
    sys.modules.setdefault("cs336_data.modal_utils", dummy_modal_utils)


_install_modal_fallback()

from cs336_basics.train_config import Config, PathsConfig  # noqa: E402
from scripts.train import train_from_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", required=True, help="tokenized 训练数据路径")
    parser.add_argument("--valid-bin", required=True, help="C4 100 domains 验证数据路径")
    parser.add_argument("--model-output", required=True, help="模型输出目录（写入 model.pt 和 step_* checkpoint）")
    args = parser.parse_args()

    # 关键：官方 scripts/train.py 用 logger.info() 输出 "Estimated validation loss"，
    # 但它没有调用 logging.basicConfig()，而 Python logging 的默认级别是 WARNING，
    # 会把 INFO 日志全部丢弃 —— 这会导致每个 eval 点的验证 loss 无法留下任何记录。
    # 这里显式把根logger 配成 INFO，确保 learning curve 能被写进日志。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    cfg = Config(
        paths=PathsConfig(
            train_bin=Path(args.train_bin),
            valid_bin=Path(args.valid_bin),
            model_output=Path(args.model_output),
        ),
    )

    train_from_config(cfg)


if __name__ == "__main__":
    main()
