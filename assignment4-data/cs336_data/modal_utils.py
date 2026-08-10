from pathlib import Path, PurePosixPath

from cs336_data.common import MODAL_SHARED_PATH

SUNET_ID = "TODO"  # NOTE: modal_utils.py should remain effectively unchanged other than adding your SUNET_ID

# ---------------------------------------------------------------------------
# 本地（非 modal）跑训练时，不强制依赖 modal 库，也无需 SUNET_ID。
# 这样 scripts/train.py 里 `from cs336_data.modal_utils import app, build_image,
# VOLUME_MOUNTS` 的顶层 import 在本地 torchrun 下也能通过。
# 只有真正用 `uv run modal run ...` 跑云上任务时才需要 modal 和 SUNET_ID。
# ---------------------------------------------------------------------------
try:
    import modal

    _HAS_MODAL = True
except Exception:  # noqa: BLE001  - 没有 modal 库时本地跑也要能 import
    _HAS_MODAL = False

if _HAS_MODAL and SUNET_ID != "TODO":
    (DATA_PATH := Path("data")).mkdir(exist_ok=True)

    app = modal.App(f"data-{SUNET_ID}")
    data_volume = modal.Volume.from_name(f"data-{SUNET_ID}", create_if_missing=True, version=2)
    shared_data_volume = modal.Volume.from_name(
        "a4-shared-data", create_if_missing=True, version=2, environment_name="cs336-shared-data"
    )

    def build_image(*, include_tests: bool = False) -> modal.Image:
        image = modal.Image.debian_slim(python_version="3.12")
        image = image.uv_sync()
        image = image.add_local_python_source("cs336_basics")
        image = image.add_local_python_source("cs336_data")
        image = image.add_local_file("AGENTS.md", "/root/AGENTS.md")
        image = image.add_local_file("CLAUDE.md", "/root/CLAUDE.md")
        if include_tests:
            image = image.add_local_dir("tests", remote_path="/root/tests")
        return image

    VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
        "/root/data": data_volume,
        str(MODAL_SHARED_PATH): shared_data_volume.read_only(),
    }

    MODAL_SECRETS = []
else:
    # ---- 本地模式：提供兼容的 dummy app / build_image / VOLUME_MOUNTS / MODAL_SECRETS ----
    # scripts/train.py 顶层用 @app.function(...) 和 @app.local_entrypoint() 装饰，
    # 所以 dummy app 需要带 function() 和 local_entrypoint() 两个可调用方法。
    class _DummyApp:
        def function(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

        def local_entrypoint(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco

    app = _DummyApp()

    def build_image(*, include_tests: bool = False):
        return None

    VOLUME_MOUNTS = {}
    MODAL_SECRETS = []
