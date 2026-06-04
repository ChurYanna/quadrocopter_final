from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisualFrameRecord:
    """One saved UAV-front RGB observation frame for later VLM analysis."""

    path: str
    label: str
    reason: str
    sim_time: float
    width: int
    height: int
    render_mode: str = 'uav_front_rgb'
    camera_name: str | None = None
    source_uav_id: int | None = None
    selection_policy: str = 'frontmost_uav'
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'path': self.path,
            'label': self.label,
            'reason': self.reason,
            'sim_time': float(self.sim_time),
            'width': int(self.width),
            'height': int(self.height),
            'render_mode': self.render_mode,
            'camera_name': self.camera_name,
            'source_uav_id': self.source_uav_id,
            'selection_policy': self.selection_policy,
            'error': self.error,
        }


class VisualFrameCapture:
    """Event-triggered UAV-front RGB keyframe sampler.

    This module opens the real image-input side for future VLM analysis.  It
    records raw RGB frames from an onboard/front camera, but never changes
    control targets or planning decisions.
    """

    def __init__(
        self,
        output_dir: str | Path,
        width: int = 640,
        height: int = 360,
        min_interval: float = 2.5,
        camera_name: str = 'frontmost_uav',
        camera_names: tuple[str, ...] | list[str] | None = None,
        enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.width = int(width)
        self.height = int(height)
        self.min_interval = float(min_interval)
        self.camera_name = str(camera_name)
        self.camera_names = tuple(camera_names or tuple(f'uav{index}_front_rgb' for index in range(5)))
        self.enabled = bool(enabled)
        self._last_capture_time: float | None = None
        self._renderer = None
        self._render_failed = False
        self._last_error: str | None = None
        self.last_frame_record: VisualFrameRecord | None = None
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def capture(
        self,
        model: Any,
        data: Any,
        sim_time: float,
        label: str,
        reason: str,
        force: bool = False,
    ) -> VisualFrameRecord | None:
        if not self.enabled:
            return None
        sim_time = float(sim_time)
        if (
            not force
            and self._last_capture_time is not None
            and sim_time - self._last_capture_time < self.min_interval
        ):
            return None
        safe_label = self._safe_label(label)
        path = self.output_dir / f'{sim_time:08.2f}_{safe_label}.png'
        if self._render_failed:
            selected_camera, source_uav_id = self._select_camera(model, data)
            record = VisualFrameRecord(
                path=str(path),
                label=str(label),
                reason=str(reason),
                sim_time=sim_time,
                width=self.width,
                height=self.height,
                camera_name=selected_camera,
                source_uav_id=source_uav_id,
                selection_policy=self.camera_name,
                error=f'uav_front_rgb_render_disabled_after_failure: {self._last_error}',
            )
            self.last_frame_record = record
            return record
        try:
            selected_camera, source_uav_id = self._select_camera(model, data)
            image = self._render_front_rgb(model, data, selected_camera)
            self._save_image(path, image)
            self._last_capture_time = sim_time
            record = VisualFrameRecord(
                path=str(path),
                label=str(label),
                reason=str(reason),
                sim_time=sim_time,
                width=self.width,
                height=self.height,
                camera_name=selected_camera,
                source_uav_id=source_uav_id,
                selection_policy=self.camera_name,
            )
            self.last_frame_record = record
            return record
        except Exception as exc:
            self._last_capture_time = sim_time
            self._render_failed = True
            self._last_error = str(exc)
            self.close()
            selected_camera, source_uav_id = self._safe_selected_camera(model, data)
            record = VisualFrameRecord(
                path=str(path),
                label=str(label),
                reason=str(reason),
                sim_time=sim_time,
                width=self.width,
                height=self.height,
                camera_name=selected_camera,
                source_uav_id=source_uav_id,
                selection_policy=self.camera_name,
                error=str(exc),
            )
            self.last_frame_record = record
            return record

    def _select_camera(self, model: Any, data: Any) -> tuple[str, int | None]:
        import mujoco

        if self.camera_name != 'frontmost_uav':
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name) < 0:
                raise ValueError(f'camera not found: {self.camera_name}')
            return self.camera_name, self._camera_name_to_uav_id(self.camera_name)

        candidates: list[tuple[float, int, str]] = []
        for uav_id, camera_name in enumerate(self.camera_names):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                continue
            qpos_index = 7 * uav_id
            if qpos_index >= len(data.qpos):
                continue
            candidates.append((float(data.qpos[qpos_index]), int(uav_id), str(camera_name)))
        if not candidates:
            raise ValueError('no UAV front RGB camera is available')
        _, source_uav_id, selected_camera = max(candidates, key=lambda item: item[0])
        return selected_camera, source_uav_id

    def _safe_selected_camera(self, model: Any, data: Any) -> tuple[str, int | None]:
        try:
            return self._select_camera(model, data)
        except Exception:
            fallback = self.camera_names[0] if self.camera_names else self.camera_name
            return str(fallback), self._camera_name_to_uav_id(str(fallback))

    def _render_front_rgb(self, model: Any, data: Any, camera_name: str):
        import mujoco

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
            raise ValueError(f'camera not found: {camera_name}')
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._renderer.update_scene(data, camera=camera_name)
        return self._renderer.render()

    @staticmethod
    def _camera_name_to_uav_id(camera_name: str) -> int | None:
        match = re.match(r'uav(\d+)_front_rgb$', str(camera_name))
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _save_image(path: Path, image) -> None:
        import matplotlib.image as mpimg

        path.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(path, image)

    @staticmethod
    def _safe_label(label: str) -> str:
        text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(label).strip())
        return text.strip('_') or 'visual_frame'
