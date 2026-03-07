"""
skeletonize.py — Stage 4: Mask → Skeleton → NetworkX Graph
============================================================
Converts instance segmentation masks into 1-pixel-wide skeletons
and builds a graph where:
  - Nodes = junction points + endpoints of the skeleton
  - Edges = fiber segments between nodes

Then computes:
  - Length (pixels → micrometers)
  - Tortuosity (path_length / euclidean_distance)
  - Branching count (number of junction nodes)

Also computes pseudo-3D depth estimation from pixel intensity
profile along the skeleton path (Beer-Lambert law).

Usage:
    from quantification.skeletonize import mask_to_metrics
    metrics = mask_to_metrics(binary_mask, intensity_image)
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import label as nd_label
import networkx as nx
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask to a 1-pixel-wide skeleton.

    Args:
        binary_mask: (H, W) uint8 or bool mask

    Returns:
        skeleton: (H, W) bool skeleton
    """
    # Morphological cleanup before skeletonize
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(
        binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2
    )
    skel = skeletonize(cleaned > 0)
    return skel


def find_skeleton_nodes(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find junction points (degree>2) and endpoints (degree=1) in skeleton.

    Args:
        skeleton: (H, W) bool skeleton

    Returns:
        junctions: (N, 2) array of [y, x] junction coordinates
        endpoints:  (M, 2) array of [y, x] endpoint coordinates
    """
    # Count neighbors for each skeleton pixel using 8-connectivity
    skel_int = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # Don't count center pixel itself
    neighbor_count = cv2.filter2D(skel_int, -1, kernel)

    # On skeleton: neighbor_count gives degree
    on_skeleton = skeleton
    degree = neighbor_count * on_skeleton

    # Junction: degree >= 3 (branching point)
    junction_mask = (degree >= 3) & on_skeleton
    # Endpoint: degree == 1
    endpoint_mask = (degree == 1) & on_skeleton
    # Also include isolated pixels (degree == 0 but on skeleton)
    isolated_mask = (degree == 0) & on_skeleton

    junctions = np.argwhere(junction_mask)    # [[y, x], ...]
    endpoints  = np.argwhere(endpoint_mask | isolated_mask)

    return junctions, endpoints


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Convert a skeleton to a NetworkX graph.

    Algorithm:
      1. Find all junction + endpoint nodes
      2. For each pair of adjacent skeleton pixels, add an edge by tracing
         the skeleton between nodes

    Args:
        skeleton: (H, W) bool skeleton

    Returns:
        G: NetworkX Graph
           node attr: "type" = "junction", "endpoint", or "skeleton"
           edge attr: "path" = list of (y,x) coordinates along segment
                      "pixel_length" = number of pixels in segment
    """
    G = nx.Graph()
    junctions, endpoints = find_skeleton_nodes(skeleton)

    # Mark all special nodes
    special_pts = set()
    node_types = {}
    for pt in junctions:
        key = (int(pt[0]), int(pt[1]))
        special_pts.add(key)
        node_types[key] = "junction"
    for pt in endpoints:
        key = (int(pt[0]), int(pt[1]))
        special_pts.add(key)
        node_types[key] = "endpoint"

    # Add nodes with attributes
    for key, ntype in node_types.items():
        G.add_node(key, y=key[0], x=key[1], node_type=ntype)

    # Trace skeleton segments between nodes
    visited = np.zeros_like(skeleton, dtype=bool)

    def get_skeleton_neighbors(y, x):
        """8-connected neighbors that are on the skeleton."""
        nbrs = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and
                    0 <= nx_ < skeleton.shape[1] and
                    skeleton[ny, nx_]):
                    nbrs.append((ny, nx_))
        return nbrs

    def trace_segment(start_y, start_x, prev_y, prev_x):
        """
        Trace a skeleton segment from a start node until hitting
        another node or a dead end.
        """
        path = [(start_y, start_x)]
        cy, cx = start_y, start_x
        py, px = prev_y, prev_x

        while True:
            nbrs = [n for n in get_skeleton_neighbors(cy, cx)
                    if not (n[0] == py and n[1] == px)]

            if not nbrs:
                break

            # Follow unvisited neighbors (prefer unvisited)
            unvisited = [(ny, nx_) for ny, nx_ in nbrs
                         if not visited[ny, nx_] or (ny, nx_) in special_pts]

            if not unvisited:
                break

            # Move to next pixel
            ny, nx_ = unvisited[0]
            path.append((ny, nx_))
            visited[ny, nx_] = True

            if (ny, nx_) in special_pts:
                return path, (ny, nx_)  # Hit another node

            py, px = cy, cx    # Save current as previous
            cy, cx = ny, nx_   # Update current to new pixel

        return path, None

    # Trace from each node
    for start_node in list(special_pts):
        visited[start_node[0], start_node[1]] = True
        nbrs = get_skeleton_neighbors(start_node[0], start_node[1])

        for ny, nx_ in nbrs:
            if not visited[ny, nx_] or (ny, nx_) in special_pts:
                path, end_node = trace_segment(ny, nx_, start_node[0], start_node[1])

                if end_node is not None and end_node in G.nodes:
                    if not G.has_edge(start_node, end_node):
                        G.add_edge(
                            start_node, end_node,
                            path=path,
                            pixel_length=len(path),
                        )

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_metrics(
    G: nx.Graph,
    intensity_image: Optional[np.ndarray] = None,
    px_to_um: float = 0.25,   # Default: 0.25 μm/pixel at 40× objective
) -> List[Dict]:
    """
    Compute morphological metrics for each edge in the skeleton graph.

    Args:
        G:               NetworkX skeleton graph
        intensity_image: (H, W) uint8 grayscale — for depth estimation
        px_to_um:        Pixels per micrometer (calibrate per microscope)

    Returns:
        edge_metrics: List of dicts, one per edge
    """
    metrics = []

    for u, v, data in G.edges(data=True):
        path = data.get("path", [u, v])
        pixel_length = len(path)

        # ── Length ───────────────────────────────────────────────────
        length_um = pixel_length * px_to_um

        # ── Tortuosity ────────────────────────────────────────────────
        if len(path) >= 2:
            start = np.array([path[0][0],  path[0][1]])
            end   = np.array([path[-1][0], path[-1][1]])
            euclidean_dist = np.linalg.norm(end - start) * px_to_um
            
            # Prevent extreme tortuosity on closed loops (start ≈ end)
            if euclidean_dist < 1e-3:
                tortuosity = 1.0
            else:
                tortuosity = length_um / euclidean_dist
        else:
            tortuosity = 1.0

        # ── Mean intensity along skeleton ────────────────────────────
        mean_intensity = 0.0
        if intensity_image is not None and len(path) > 0:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ys = np.clip(ys, 0, intensity_image.shape[0] - 1)
            xs = np.clip(xs, 0, intensity_image.shape[1] - 1)
            mean_intensity = float(intensity_image[ys, xs].mean())

        # ── Pseudo-3D Depth Estimation (Beer-Lambert) ─────────────────
        # I = I0 * exp(-mu * d)  →  d = -ln(I/I0) / mu
        # Approximation: I0 = background (235), mu = 0.02 (assumed)
        estimated_depth_um = 0.0
        if mean_intensity > 0 and mean_intensity < 235:
            I0 = 235.0
            mu = 0.02   # Absorption coefficient (empirical)
            ratio = max(mean_intensity / I0, 1e-6)
            estimated_depth_um = -np.log(ratio) / mu

        metrics.append({
            "edge":             (u, v),
            "pixel_length":     pixel_length,
            "length_um":        round(length_um, 3),
            "tortuosity":       round(tortuosity, 4),
            "mean_intensity":   round(mean_intensity, 2),
            "depth_um":         round(estimated_depth_um, 3),
        })

    return metrics


def compute_graph_metrics(G: nx.Graph, edge_metrics: List[Dict]) -> Dict:
    """
    Compute image-level aggregate metrics from skeleton graph.

    Args:
        G:            Full NetworkX skeleton graph
        edge_metrics: List of per-edge metric dicts

    Returns:
        graph_metrics: Dict with aggregate fibril statistics
    """
    if not edge_metrics:
        return {
            "total_fibrils": 0,
            "total_length_um": 0.0,
            "mean_length_um": 0.0,
            "mean_tortuosity": 0.0,
            "branching_points": 0,
            "fibrillation_index": 0.0,
        }

    lengths = [m["length_um"] for m in edge_metrics]
    tortuosities = [m["tortuosity"] for m in edge_metrics]

    # Count junction nodes = branching points
    branching_pts = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "junction"
    )

    # Fibrillation Index (simplified image-based version):
    # FI = (total fibril length) / (main fiber perimeter) × 100
    # Here: ratio of total skeleton length to minimum skeleton segment
    total_length = sum(lengths)
    base_length = max(min(lengths), 1.0) if lengths else 1.0
    fi = (total_length / base_length) * 10.0  # Scaled FI

    return {
        "total_fibrils":    len(edge_metrics),
        "total_length_um":  round(total_length, 3),
        "mean_length_um":   round(np.mean(lengths), 3),
        "std_length_um":    round(float(np.std(lengths)), 3),
        "mean_tortuosity":  round(np.mean(tortuosities), 4),
        "branching_points": branching_pts,
        "fibrillation_index": round(fi, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_metrics(
    binary_mask: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    px_to_um: float = 0.25,
    fibril_id: int = 0,
) -> Dict:
    """
    Full pipeline: binary mask → skeleton → graph → metrics dict.

    Args:
        binary_mask: (H, W) uint8 or bool instance mask
        intensity_image: (H, W) uint8 grayscale original image (for depth)
        px_to_um: Micrometer per pixel calibration
        fibril_id: Instance ID for output labeling

    Returns:
        metrics: {fibril_id, length_um, tortuosity, branching_count,
                  estimated_depth_um, fibrillation_index, ...}
    """
    skeleton = extract_skeleton(binary_mask)

    # If skeleton is empty — return zero metrics
    if skeleton.sum() == 0:
        return {
            "fibril_id": fibril_id,
            "length_um": 0.0,
            "tortuosity": 1.0,
            "branching_count": 0,
            "mean_intensity": 0.0,
            "estimated_depth_um": 0.0,
            "fibrillation_index": 0.0,
        }

    G = skeleton_to_graph(skeleton)
    edge_metrics = compute_edge_metrics(G, intensity_image, px_to_um)
    graph_metrics = compute_graph_metrics(G, edge_metrics)

    # Aggregate across edges for this one instance
    lengths = [m["length_um"] for m in edge_metrics]
    tortuosities = [m["tortuosity"] for m in edge_metrics]
    depths = [m["depth_um"] for m in edge_metrics]
    intensities = [m["mean_intensity"] for m in edge_metrics]

    return {
        "fibril_id":          fibril_id,
        "total_length_um":    graph_metrics["total_length_um"],
        "mean_segment_um":    graph_metrics["mean_length_um"],
        "tortuosity":         round(float(np.mean(tortuosities)), 4) if tortuosities else 1.0,
        "branching_count":    graph_metrics["branching_points"],
        "mean_intensity":     round(float(np.mean(intensities)), 2) if intensities else 0.0,
        "estimated_depth_um": round(float(np.mean(depths)), 3) if depths else 0.0,
        "fibrillation_index": graph_metrics["fibrillation_index"],
        "num_segments":       len(edge_metrics),
    }
