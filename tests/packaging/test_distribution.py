import os
import sys
import shutil
import tarfile
import zipfile
import subprocess
from pathlib import Path, PurePosixPath

import pytest


pytestmark = pytest.mark.packaging
PROJECT_ROOT = Path(__file__).parents[2].resolve()
# Conservative budgets for installation under the Reachy SDK's Windows HF cache.
WINDOWS_PATH_BUDGET = 130
WINDOWS_WHEEL_PATH_BUDGET = 71


def _git_tracked_files(project_root: Path) -> list[Path]:
    """Return git-tracked files that still exist in the working tree."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"Source packaging requires a Git checkout: {exc}")

    tracked_files = [project_root / relative_path for relative_path in result.stdout.splitlines() if relative_path]
    return [path for path in tracked_files if path.is_file()]


def test_project_file_paths_stay_within_windows_budget() -> None:
    """Git-tracked project file paths should stay below the agreed Windows budget."""
    project_root = PROJECT_ROOT
    project_files = _git_tracked_files(project_root)

    violations = []
    for path in project_files:
        relative = str(Path(project_root.name) / path.relative_to(project_root))
        length = len(relative)
        if length > WINDOWS_PATH_BUDGET:
            violations.append(
                f"Windows path budget exceeded ({WINDOWS_PATH_BUDGET}): {relative} is {length} characters long"
            )

    assert not violations, "\n".join(violations)


def test_source_distribution_installs_without_build_tools(tmp_path: Path) -> None:
    """Source distributions install offline without Node and contain complete, portable wheels."""
    project_root = PROJECT_ROOT
    source_checkout = tmp_path / "checkout"
    dist_dir = tmp_path / "dist"
    uv = shutil.which("uv")
    assert uv is not None

    for source_file in _git_tracked_files(project_root):
        target_file = source_checkout / source_file.relative_to(project_root)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    assert not (source_checkout / "src/reachy_mini_conversation_app/gen").exists()
    assert not (source_checkout / "src/reachy_mini_conversation_app/static").exists()
    assert not (source_checkout / "frontend/src/gen").exists()

    try:
        subprocess.run(
            [uv, "build", "--out-dir", str(dist_dir)],
            cwd=source_checkout,
            check=True,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        details = exc.stderr if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        pytest.fail(f"Distribution build failed while checking Windows path budget: {details}")

    source_archives = list(dist_dir.glob("*.tar.gz"))
    assert len(source_archives) == 1
    assert len(list(dist_dir.glob("*.whl"))) == 1
    extracted = tmp_path / "extracted"
    with tarfile.open(source_archives[0]) as archive:
        source_paths = [member.name.partition("/")[2] for member in archive.getmembers() if member.isfile()]
        archive.extractall(extracted, filter="data")
    long_source_paths = [path for path in source_paths if len(f"{project_root.name}/{path}") > WINDOWS_PATH_BUDGET]
    assert not long_source_paths, f"Source files exceed the SDK Windows cache path budget: {long_source_paths}"
    source_distribution = next(extracted.iterdir())
    assert (source_distribution / "PKG-INFO").is_file()
    assert (source_distribution / "README.md").read_bytes() == (project_root / "README.md").read_bytes()
    assert (source_distribution / "docs/assets/conversation_app_arch.svg").is_file()

    node_free_dist = tmp_path / "node-free-dist"
    subprocess.run(
        [uv, "build", "--wheel", "--offline", "--python", sys.executable, "--out-dir", str(node_free_dist)],
        cwd=source_distribution,
        env={**os.environ, "PATH": ""},
        check=True,
        timeout=60,
    )
    wheel_files = list(node_free_dist.glob("*.whl"))
    assert len(wheel_files) == 1

    installed = tmp_path / "installed"
    (tmp_path / "profiles").mkdir()
    with zipfile.ZipFile(wheel_files[0]) as archive:
        archived_paths = [PurePosixPath(info.filename) for info in archive.infolist() if not info.is_dir()]
        archive.extractall(installed)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "from importlib.resources import files; "
            "from reachy_mini_conversation_app.gen.reachy.conversation.v1 import api_pb, api_connect; "
            "from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY; "
            "from reachy_mini_conversation_app.profile_store import read_profile; "
            "assert DEFAULT_PROFILES_DIRECTORY == Path.cwd() / 'reachy_talk_data/profiles'; "
            "assert read_profile(None).instructions; "
            "assert Path(api_pb.__file__).is_relative_to(Path.cwd()); "
            "assert Path(api_connect.__file__).is_relative_to(Path.cwd()); "
            "assert files('reachy_mini_conversation_app').joinpath('static/index.html').is_file(); "
            "assert files('reachy_talk_data').joinpath('profiles/default/profile.md').is_file()",
        ],
        cwd=installed,
        env={**os.environ, "PATH": "", "PYTHONPATH": str(installed)},
        check=True,
        timeout=30,
    )

    for asset in (source_checkout / "src" / "reachy_mini_conversation_app" / "static").rglob("*"):
        if asset.is_file():
            assert PurePosixPath(asset.relative_to(source_checkout / "src").as_posix()) in archived_paths

    violations = []
    for path in archived_paths:
        length = len(path.as_posix())
        if length > WINDOWS_WHEEL_PATH_BUDGET:
            violations.append(
                f"Windows wheel path budget exceeded ({WINDOWS_WHEEL_PATH_BUDGET}): "
                f"{path.as_posix()} is {length} characters long"
            )

    assert not violations, "\n".join(violations)
