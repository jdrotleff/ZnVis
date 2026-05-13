"""
ZnVis: A Zincwarecode package.
License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the Zincwarecode Project.
Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
Citation
--------
If you use this module please cite us with:

Summary
-------
Main visualizer class.
"""

import os

import znvis.cameras

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import logging
import pathlib
import typing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rich.progress import Progress

import znvis
from znvis import cameras
from znvis.rendering import Mitsuba
from znvis.visualizer.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

_PARALLEL_RENDER_STATE = {}


def _get_mesh_dict_for_frame(
    particles: typing.List[znvis.Particle],
    vector_field: typing.List[znvis.VectorField] | None,
    frame_index: int,
):
    """
    Build a mesh dictionary for a specific frame index.

    Parameters
    ----------
    particles : list[znvis.Particle]
            Particle objects to render.
    vector_field : list[znvis.VectorField] | None
            Optional vector field objects to render.
    frame_index : int
            Frame index to render.

    Returns
    -------
    dict
            Mesh dictionary compatible with the Mitsuba renderer.
    """
    mesh_dict = {}
    if vector_field is not None:
        for item in vector_field:
            idx = 0 if item.static else frame_index
            mesh = (
                item.mesh_list[idx]
                if item.mesh_list is not None
                else item.get_mesh_for_frame(frame_index)
            )
            mesh_dict[item.name] = {
                "mesh": mesh,
                "bsdf": item.mesh.material.mitsuba_bsdf,
                "material": item.mesh.o3d_material,
            }
    for item in particles:
        idx = 0 if item.static else frame_index
        mesh = (
            item.mesh_list[idx]
            if item.mesh_list is not None
            else item.get_mesh_for_frame(frame_index)
        )
        mesh_dict[item.name] = {
            "mesh": mesh,
            "bsdf": item.mesh.material.mitsuba_bsdf,
            "material": item.mesh.o3d_material,
        }
    return mesh_dict


def _render_frame_parallel_worker(frame_index: int) -> int:
    """
    Render one frame inside a worker process.

    Parameters
    ----------
    frame_index : int
            Frame index to render.

    Returns
    -------
    int
            The rendered frame index.
    """
    state = _PARALLEL_RENDER_STATE
    renderer = state.get("renderer")
    if renderer is None:
        renderer = Mitsuba()
        state["renderer"] = renderer

    mesh_dict = _get_mesh_dict_for_frame(
        particles=state["particles"],
        vector_field=state["vector_field"],
        frame_index=frame_index,
    )
    if state["camera"] is not None:
        view_matrix = state["camera"].get_view_matrix(frame_index)
    else:
        view_matrix = state["view_matrix"]

    renderer.render_mesh_objects(
        mesh_dict,
        view_matrix,
        save_dir=state["frame_folder"],
        save_name=f"frame_{frame_index:0>6}.png",
        resolution=state["renderer_resolution"],
        samples_per_pixel=state["renderer_spp"],
    )
    return frame_index


def _initialize_parallel_worker(state: dict):
    """
    Initialize per-process rendering state for parallel worker execution.

    Parameters
    ----------
    state : dict
            Worker-local rendering state used by parallel frame workers.
    """
    global _PARALLEL_RENDER_STATE
    _PARALLEL_RENDER_STATE = dict(state)
    _PARALLEL_RENDER_STATE["renderer"] = None


class Headless_Visualizer(BaseVisualizer):
    """
    Main class to perform visualization.

    Attributes
    ----------
    particles : list[znvis.Particle]
            A list of particle objects to add to the visualizer.
    counter : int
            Internally stored counter to track which configuration is currently
            being viewed.
    """

    def __init__(
        self,
        particles: typing.List[znvis.Particle],
        vector_field: typing.List[znvis.VectorField] | None = None,
        output_folder: typing.Union[str, pathlib.Path] = "./",
        frame_rate: int = 24,
        number_of_steps: int | None = None,
        keep_frames: bool = True,
        bounding_box: znvis.BoundingBox | None = None,
        video_format: str = "mp4",
        video_title: str = "ZnVis-Video",
        renderer_resolution: typing.Sequence[int] | None = None,
        renderer_spp: int = 64,
        renderer: Mitsuba | None = None,
        do_create_video: bool = True,
        camera: cameras.BaseCamera | None = None,
        parallel_render_workers: int = 1,
        parallel_render_enabled: bool = False,
    ):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
        vector_field : list[znvis.VectorField]
                List of vector fields to add to the visualizer.
        output_folder: typing.Union[str, pathlib.Path]
                Path to the output folder where the video and frames will be saved.
        frame_rate : int
                Frame rate for the visualizer measured in frames per second (fps)
        number_of_steps : int
                Number of steps in the visualization. If None, the zeroth order of one
                particle is taken. This is left as an option in case the user wishes
                to overlay two particle trajectories of different length.
        keep_frames : bool
                If True, the visualizer will keep all frames
                after combining them into a video.
        video_format : str
                The format of the video to be generated. Default is mp4.
        video_title : str
                The name of the video file.
        renderer_resolution : list
                List containing the resolution of the rendered videos and screenshots
        renderer_spp : int
                Samples per pixel for the rendered videos and screenshots.
        renderer : Mitsuba
                The renderer engine to use for rendering.
        do_create_video: bool
                If True, the visualizer will create a video from the frames.
        camera : znvis.cameras.Camera object
                Camera object to use for the visualization. If None, a default camera
                will be used.
        parallel_render_workers : int
                Number of worker processes to use when parallel rendering is enabled.
        parallel_render_enabled : bool
                If True, render frames with a process pool.
        """
        # Call parent constructor
        super().__init__(
            particles=particles,
            vector_field=vector_field,
            output_folder=output_folder,
            frame_rate=frame_rate,
            number_of_steps=number_of_steps,
            keep_frames=keep_frames,
            bounding_box=bounding_box,
            video_format=video_format,
            video_title=video_title,
            renderer_resolution=renderer_resolution,
            renderer_spp=renderer_spp,
            renderer=renderer,
            parallel_render_workers=parallel_render_workers,
            parallel_render_enabled=parallel_render_enabled,
        )

        # Headless-specific attributes
        self.do_create_video = do_create_video
        self.app = None
        self.vis = None

        # Set the camera
        self.camera = camera
        if self.camera is not None:
            self.camera.verify_camera_setup_for_rendering()
        else:
            print(
                "No camera provided, using a default view matrix."
                "Consider using the StaticCamera class to set the "
                "view matrix based on your particle positions or box size."
            )
            self.view_matrix = np.array(
                [[1, 0, 0, -100], [0, 1, 0, -90], [0, 0, 1, -230], [0, 0, 0, 1]]
            )

    def _render_frame(self, frame_index: int, renderer: Mitsuba | None = None):
        """
        Render a single frame by index.

        Parameters
        ----------
        frame_index : int
                Frame index to render.
        renderer : Mitsuba, optional
                Renderer instance to use. Defaults to the visualizer renderer.
        """
        mesh_dict = _get_mesh_dict_for_frame(
            particles=self.particles,
            vector_field=self.vector_field,
            frame_index=frame_index,
        )
        view_matrix = (
            self.camera.get_view_matrix(frame_index)
            if self.camera is not None
            else self.view_matrix
        )
        (renderer or self.renderer).render_mesh_objects(
            mesh_dict,
            view_matrix,
            save_dir=self.frame_folder,
            save_name=f"frame_{frame_index:0>6}.png",
            resolution=self.renderer_resolution,
            samples_per_pixel=self.renderer_spp,
        )

    def _render_frames_serial(self):
        """
        Render all frames sequentially in the main process.
        """
        with Progress() as progress:
            task = progress.add_task("Saving scenes...", total=self.number_of_steps)
            for frame_index in range(self.number_of_steps):
                self._render_frame(frame_index)
                progress.update(task, advance=1)

    def _render_frames_parallel(self):
        """
        Render all frames with a process pool.

        Falls back to serial rendering if parallel worker startup fails.
        """
        if self.parallel_render_workers == 1:
            self._render_frames_serial()
            return

        worker_state = {
            "particles": self.particles,
            "vector_field": self.vector_field,
            "camera": self.camera,
            "view_matrix": self.view_matrix,
            "frame_folder": self.frame_folder,
            "renderer_resolution": self.renderer_resolution,
            "renderer_spp": self.renderer_spp,
        }

        try:
            with ProcessPoolExecutor(
                max_workers=self.parallel_render_workers,
                initializer=_initialize_parallel_worker,
                initargs=(worker_state,),
            ) as executor:
                with Progress() as progress:
                    task = progress.add_task(
                        "Saving scenes...", total=self.number_of_steps
                    )
                    for _ in executor.map(
                        _render_frame_parallel_worker, range(self.number_of_steps)
                    ):
                        progress.update(task, advance=1)
        except Exception:
            logger.exception(
                "Parallel rendering failed, likely due to worker initialization "
                "or frame-state serialization. Falling back to serial rendering "
                "for this run."
            )
            self._render_frames_serial()

    def _record_trajectory(self):
        """
        Record the trajectory.
        """
        if self.parallel_render_enabled:
            self._render_frames_parallel()
        else:
            self._render_frames_serial()

        if self.do_create_video:
            self._create_movie()

    def render_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Launches the visualization.
        """
        self.frame_folder.mkdir(parents=True, exist_ok=True)
        self._record_trajectory()
