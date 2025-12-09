"""SAM 3D service for converting 2D images to 3D objects.

This module combines depth estimation with SAM 2D segmentation to create
3D meshes from single images.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .sam_annotator import SamAnnotator, SamSegment
from ..service.sam_service import SamService

LOGGER = logging.getLogger(__name__)


@dataclass
class Mesh3D:
    """3D mesh data structure."""
    
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (M, 3) int32 - triangle indices
    vertex_colors: Optional[np.ndarray] = None  # (N, 3) uint8 RGB colors
    texture_coords: Optional[np.ndarray] = None  # (N, 2) float32 UV coordinates
    
    def get_vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)
    
    def get_face_count(self) -> int:
        """Get number of faces."""
        return len(self.faces)


class Sam3DService:
    """Service for converting 2D images to 3D objects using depth estimation and SAM."""
    
    def __init__(
        self,
        sam_service: Optional[SamService] = None,
        sam_checkpoint: Optional[Path] = None,
        device: str = "cpu",
        depth_model: str = "zoedepth",
    ):
        """Initialize SAM 3D service.
        
        Args:
            sam_service: Pre-initialized SAM service (optional)
            sam_checkpoint: Path to SAM checkpoint (if sam_service not provided)
            device: Device to run inference on (cpu, cuda, mps)
            depth_model: Depth estimation model to use (zoedepth, dpt, midas)
        """
        self.device = device
        self.depth_model_type = depth_model
        self.sam_service = sam_service
        self.sam_checkpoint = sam_checkpoint
        
        self.depth_model = None
        self._depth_initialized = False
        
        LOGGER.info(
            "Created Sam3DService: depth_model=%s, device=%s",
            depth_model,
            device,
        )
    
    def initialize(self) -> None:
        """Initialize depth estimation model."""
        if self._depth_initialized:
            return
        
        try:
            if self.depth_model_type == "zoedepth":
                self._initialize_zoedepth()
            elif self.depth_model_type == "dpt":
                self._initialize_dpt()
            elif self.depth_model_type == "midas":
                self._initialize_midas()
            else:
                raise ValueError(f"Unknown depth model: {self.depth_model_type}")
            
            self._depth_initialized = True
            LOGGER.info("Depth estimation model initialized: %s", self.depth_model_type)
            
        except ImportError as exc:
            LOGGER.error(
                "Failed to import depth estimation library. "
                "Install with: pip install transformers torch"
            )
            raise
    
    def _initialize_zoedepth(self) -> None:
        """Initialize ZoeDepth model."""
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            import torch
            
            model_name = "Intel/zoedepth-nyu"
            LOGGER.info("Loading ZoeDepth model: %s", model_name)
            
            self.depth_processor = AutoImageProcessor.from_pretrained(model_name)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
        except ImportError:
            # Fallback to simpler approach if transformers not available
            LOGGER.warning("ZoeDepth requires transformers. Falling back to simple depth estimation.")
            self.depth_model_type = "simple"
            self._initialize_simple_depth()
    
    def _initialize_dpt(self) -> None:
        """Initialize DPT (Dense Prediction Transformer) model."""
        try:
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            import torch
            
            model_name = "Intel/dpt-large"
            LOGGER.info("Loading DPT model: %s", model_name)
            
            self.depth_processor = DPTImageProcessor.from_pretrained(model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
        except ImportError:
            LOGGER.warning("DPT requires transformers. Falling back to simple depth estimation.")
            self.depth_model_type = "simple"
            self._initialize_simple_depth()
    
    def _initialize_midas(self) -> None:
        """Initialize MiDaS model."""
        try:
            import torch
            import torch.hub
            
            model_type = "DPT_Large"
            LOGGER.info("Loading MiDaS model: %s", model_type)
            
            self.depth_model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            # MiDaS transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform
                
        except Exception as exc:
            LOGGER.warning("MiDaS initialization failed: %s. Falling back to simple depth.", exc)
            self.depth_model_type = "simple"
            self._initialize_simple_depth()
    
    def _initialize_simple_depth(self) -> None:
        """Initialize simple depth estimation (fallback)."""
        # Simple gradient-based depth estimation as fallback
        LOGGER.info("Using simple gradient-based depth estimation")
        self.depth_model = "simple"
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from image.
        
        Args:
            image: Input image (H, W, 3) RGB or BGR
            
        Returns:
            Depth map (H, W) float32, normalized to [0, 1]
        """
        if not self._depth_initialized:
            self.initialize()
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (OpenCV default)
            if image.dtype == np.uint8:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = image
        
        if self.depth_model_type == "zoedepth":
            return self._estimate_depth_zoedepth(image_rgb)
        elif self.depth_model_type == "dpt":
            return self._estimate_depth_dpt(image_rgb)
        elif self.depth_model_type == "midas":
            return self._estimate_depth_midas(image_rgb)
        else:
            return self._estimate_depth_simple(image_rgb)
    
    def _estimate_depth_zoedepth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using ZoeDepth."""
        import torch
        
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth = outputs.predicted_depth.cpu().numpy()[0, 0]
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    def _estimate_depth_dpt(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using DPT."""
        import torch
        
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            depth = outputs.predicted_depth.cpu().numpy()[0, 0]
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    def _estimate_depth_midas(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS."""
        import torch
        
        # Prepare input
        input_batch = self.depth_transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            depth = prediction.cpu().numpy()[0, 0]
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    def _estimate_depth_simple(self, image: np.ndarray) -> np.ndarray:
        """Simple gradient-based depth estimation (fallback)."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Convert to depth (inverse: high gradient = close, low gradient = far)
        depth = 1.0 / (1.0 + gradient_magnitude / gradient_magnitude.max())
        
        # Apply Gaussian blur for smoothness
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.astype(np.float32)
    
    def create_mesh_from_depth(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        segments: Optional[list[SamSegment]] = None,
        depth_scale: float = 1.0,
        mesh_resolution: Optional[int] = None,
    ) -> Mesh3D:
        """Create 3D mesh from depth map and optional segments.
        
        Args:
            image: Original image (H, W, 3) RGB
            depth_map: Depth map (H, W) float32 [0, 1]
            segments: Optional SAM segments to focus on specific regions
            depth_scale: Scale factor for depth values
            mesh_resolution: Target resolution for mesh (None = use image size)
            
        Returns:
            Mesh3D object with vertices and faces
        """
        h, w = depth_map.shape[:2]
        
        # Downsample if mesh_resolution specified
        if mesh_resolution and mesh_resolution < min(h, w):
            scale = mesh_resolution / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            depth_map = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        
        # Create vertex grid
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Normalize coordinates to [-1, 1] range
        x_norm = (x_coords / w) * 2.0 - 1.0
        y_norm = (y_coords / h) * 2.0 - 1.0
        
        # Use depth as z coordinate (scale and center)
        z_coords = (depth_map - 0.5) * depth_scale
        
        # Stack to create vertices (N, 3)
        vertices = np.stack([x_norm, y_norm, z_coords], axis=-1).reshape(-1, 3).astype(np.float32)
        
        # Create vertex colors from image
        if len(image.shape) == 3:
            vertex_colors = image.reshape(-1, 3).astype(np.uint8)
        else:
            vertex_colors = None
        
        # Create faces (triangles) for quad mesh
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                # Current quad indices
                idx = i * w + j
                idx_right = i * w + (j + 1)
                idx_down = (i + 1) * w + j
                idx_down_right = (i + 1) * w + (j + 1)
                
                # Two triangles per quad
                faces.append([idx, idx_right, idx_down])
                faces.append([idx_right, idx_down_right, idx_down])
        
        faces = np.array(faces, dtype=np.int32)
        
        # Apply segment masks if provided
        if segments:
            vertices, faces, vertex_colors = self._apply_segment_masks(
                vertices, faces, vertex_colors, segments, w, h
            )
        
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
        )
    
    def _apply_segment_masks(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_colors: Optional[np.ndarray],
        segments: list[SamSegment],
        img_width: int,
        img_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Filter mesh to only include regions covered by segments."""
        # Create combined mask from all segments
        combined_mask = np.zeros((img_height, img_width), dtype=bool)
        for segment in segments:
            mask_resized = cv2.resize(
                segment.mask.astype(np.uint8),
                (img_width, img_height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            combined_mask |= mask_resized
        
        # Filter vertices and remap indices
        vertex_mask = combined_mask.reshape(-1)
        vertex_indices = np.where(vertex_mask)[0]
        vertex_map = np.full(len(vertex_mask), -1, dtype=np.int32)
        vertex_map[vertex_indices] = np.arange(len(vertex_indices))
        
        # Filter vertices
        filtered_vertices = vertices[vertex_indices]
        filtered_colors = vertex_colors[vertex_indices] if vertex_colors is not None else None
        
        # Filter faces (only keep faces where all vertices are in mask)
        filtered_faces = []
        for face in faces:
            if all(vertex_mask[v] for v in face):
                mapped_face = [vertex_map[v] for v in face]
                filtered_faces.append(mapped_face)
        
        filtered_faces = np.array(filtered_faces, dtype=np.int32) if filtered_faces else np.array([], dtype=np.int32)
        
        return filtered_vertices, filtered_faces, filtered_colors
    
    def convert_image_to_3d(
        self,
        image_path: Path | str,
        output_dir: Path | str,
        model_id: str,
        use_sam: bool = True,
        depth_scale: float = 1.0,
        mesh_resolution: Optional[int] = None,
    ) -> dict:
        """Convert image to 3D mesh.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save output files
            model_id: Unique identifier for this model
            use_sam: Whether to use SAM segmentation to focus on objects
            depth_scale: Scale factor for depth values
            mesh_resolution: Target mesh resolution (None = use image size)
            
        Returns:
            Dictionary with metadata about the generated mesh
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        LOGGER.info("Converting image to 3D: %s (%dx%d)", image_path.name, w, h)
        
        # Estimate depth
        LOGGER.info("Estimating depth map...")
        depth_map = self.estimate_depth(image_rgb)
        
        # Get SAM segments if requested
        segments = None
        if use_sam and self.sam_service:
            if not self.sam_service.is_ready():
                # Initialize SAM service if needed
                if self.sam_checkpoint:
                    self.sam_service.initialize(Path(self.sam_checkpoint))
            
            if self.sam_service.is_ready():
                LOGGER.info("Running SAM segmentation...")
                segments = self.sam_service.segment_frame(image_path, use_cache=True)
                LOGGER.info("Found %d segments", len(segments))
        
        # Create mesh
        LOGGER.info("Creating 3D mesh...")
        mesh = self.create_mesh_from_depth(
            image_rgb,
            depth_map,
            segments=segments,
            depth_scale=depth_scale,
            mesh_resolution=mesh_resolution,
        )
        
        # Save mesh files
        obj_path = output_dir / f"{model_id}_mesh.obj"
        glb_path = output_dir / f"{model_id}_mesh.glb"
        depth_map_path = output_dir / f"{model_id}_depth.png"
        metadata_path = output_dir / f"{model_id}_metadata.json"
        
        # Save OBJ file
        self._save_obj(mesh, obj_path, image_path.name)
        
        # Save GLB file (if pygltflib available)
        try:
            self._save_glb(mesh, glb_path)
        except ImportError:
            LOGGER.warning("pygltflib not available, skipping GLB export")
        
        # Save depth map for visualization
        depth_vis = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(str(depth_map_path), depth_vis)
        
        # Save metadata
        metadata = {
            "model_id": model_id,
            "source_image": str(image_path),
            "image_size": {"width": w, "height": h},
            "mesh_stats": {
                "vertex_count": mesh.get_vertex_count(),
                "face_count": mesh.get_face_count(),
            },
            "depth_model": self.depth_model_type,
            "used_sam": use_sam and segments is not None,
            "segment_count": len(segments) if segments else 0,
            "depth_scale": depth_scale,
            "mesh_resolution": mesh_resolution or min(w, h),
            "output_files": {
                "obj": str(obj_path),
                "glb": str(glb_path) if glb_path.exists() else None,
                "depth_map": str(depth_map_path),
            },
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        LOGGER.info(
            "3D conversion complete: %d vertices, %d faces",
            mesh.get_vertex_count(),
            mesh.get_face_count(),
        )
        
        return metadata
    
    def _save_obj(self, mesh: Mesh3D, output_path: Path, texture_name: Optional[str] = None) -> None:
        """Save mesh as OBJ file."""
        with open(output_path, "w") as f:
            f.write("# OBJ file generated by SAM 3D Service\n")
            
            # Write vertices
            for vertex in mesh.vertices:
                f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Write vertex colors (as comments or separate file)
            if mesh.vertex_colors is not None:
                for i, color in enumerate(mesh.vertex_colors):
                    f.write(f"# vc {i+1} {color[0]} {color[1]} {color[2]}\n")
            
            # Write faces (1-indexed in OBJ format)
            for face in mesh.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def _save_glb(self, mesh: Mesh3D, output_path: Path) -> None:
        """Save mesh as GLB file using trimesh."""
        try:
            import trimesh
        except ImportError:
            LOGGER.warning("trimesh not available, skipping GLB export")
            return
        
        # Create trimesh object
        tri_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=mesh.vertex_colors,
        )
        
        # Export as GLB
        tri_mesh.export(str(output_path), file_type="glb")
        LOGGER.info("Saved GLB file: %s", output_path)


__all__ = ["Sam3DService", "Mesh3D"]

