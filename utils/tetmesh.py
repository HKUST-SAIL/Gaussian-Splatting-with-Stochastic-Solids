### adapted from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py

# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import tempfile

import numpy as np
import torch
import triton
import triton.language as tl

__all__ = ["marching_tetrahedra"]


class InsufficientMarchingMemory(RuntimeError):
    """The resident edge sort would exceed the available CUDA memory."""

triangle_table = torch.tensor(
    [
        [-1, -1, -1, -1, -1, -1],
        [1, 0, 2, -1, -1, -1],
        [4, 0, 3, -1, -1, -1],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, -1, -1, -1],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, -1, -1, -1],
        [4, 5, 2, -1, -1, -1],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, -1, -1, -1],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, -1, -1, -1],
        [2, 0, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ],
    dtype=torch.long,
)

num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long)
base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)
v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


@triton.jit
def _count_surface_triangles(tets, sdf, valids, triangle_counts, count_table, n: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    inside = row < n
    a = tl.load(tets + 4 * row, inside, other=0)
    b = tl.load(tets + 4 * row + 1, inside, other=0)
    c = tl.load(tets + 4 * row + 2, inside, other=0)
    d = tl.load(tets + 4 * row + 3, inside, other=0)
    case = (tl.load(sdf + a) > 0).to(tl.int32)
    case += 2 * (tl.load(sdf + b) > 0).to(tl.int32)
    case += 4 * (tl.load(sdf + c) > 0).to(tl.int32)
    case += 8 * (tl.load(sdf + d) > 0).to(tl.int32)
    valid = tl.load(valids + a) & tl.load(valids + b)
    valid &= tl.load(valids + c) & tl.load(valids + d)
    count = tl.where(valid, tl.load(count_table + case), 0)
    tl.store(triangle_counts + row, count, inside)


@triton.jit
def _emit_surface_edge_keys(tets, sdf, triangle_counts, offsets, triangle_edges, base_edges,
                            out_keys, n: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    inside = row < n
    a = tl.load(tets + 4 * row, inside, other=0)
    b = tl.load(tets + 4 * row + 1, inside, other=0)
    c = tl.load(tets + 4 * row + 2, inside, other=0)
    d = tl.load(tets + 4 * row + 3, inside, other=0)
    case = (tl.load(sdf + a) > 0).to(tl.int32)
    case += 2 * (tl.load(sdf + b) > 0).to(tl.int32)
    case += 4 * (tl.load(sdf + c) > 0).to(tl.int32)
    case += 8 * (tl.load(sdf + d) > 0).to(tl.int32)
    count = tl.load(triangle_counts + row, inside, other=0)
    start = tl.load(offsets + row, inside, other=0) - count
    for slot in range(6):
        write = inside & (slot < 3 * count)
        edge = tl.load(triangle_edges + 6 * case + slot, write, other=0)
        first = tl.load(base_edges + 2 * edge, write, other=0)
        second = tl.load(base_edges + 2 * edge + 1, write, other=0)
        lo_vertex = tl.where(first == 0, a, tl.where(first == 1, b, tl.where(first == 2, c, d)))
        hi_vertex = tl.where(second == 0, a, tl.where(second == 1, b, tl.where(second == 2, c, d)))
        lo = tl.minimum(lo_vertex, hi_vertex).to(tl.uint64)
        hi = tl.maximum(lo_vertex, hi_vertex).to(tl.uint64)
        key = (lo << 32) | hi
        tl.store(out_keys + 3 * start + slot, key.to(tl.int64), write)


@torch.no_grad()
def marching_tetrahedra(vertices, tets, sdf, scales, valids, chunk_size=262144):
    """Extract a single mesh with bounded CUDA workspace.

    GPU tetrahedra remain resident on the device through edge deduplication.
    CPU tetrahedra use a bounded streaming path and deduplicate once on the
    host. Output faces and edge IDs follow the tetrahedra device; endpoint
    data follows ``vertices.device``.
    """
    if not vertices.is_cuda or not sdf.is_cuda or not valids.is_cuda:
        raise ValueError("streaming marching tetrahedra requires CUDA vertex, SDF and validity tensors")
    if tets.dtype != torch.int32 or tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must be an int32 tensor of shape [N, 4]")
    if vertices.shape[0] >= 2**31:
        raise ValueError("int32 tetrahedron indices require fewer than 2^31 vertices")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = vertices.device
    sdf = sdf.contiguous()
    valids = valids.contiguous()
    count_table = num_triangles_table.to(device=device, dtype=torch.int32)
    face_table = triangle_table.to(device=device, dtype=torch.int32).contiguous()
    edges_table = base_tet_edges.to(device=device, dtype=torch.int32)

    if tets.is_cuda:
        if tets.device != device:
            raise ValueError("tets and vertices must be on the same CUDA device")
        tets = tets.contiguous()
        face_count = 0
        for start in range(0, tets.shape[0], chunk_size):
            tet_chunk = tets[start:start + chunk_size]
            n = tet_chunk.shape[0]
            counts = torch.empty(n, device=device, dtype=torch.int32)
            _count_surface_triangles[(triton.cdiv(n, 256),)](
                tet_chunk, sdf, valids, counts, count_table, n, 256)
            face_count += int(counts.sum().item())

        if face_count == 0:
            ids = torch.empty((0, 2), dtype=torch.int32, device=device)
            faces = torch.empty((0, 3), dtype=torch.int32, device=device)
        else:
            # torch.unique needs sorting workspace in addition to the key
            # array and inverse map. Fail early with a useful CPU option.
            key_bytes = face_count * 3 * 8
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < 7 * key_bytes + 2 * 2**30:
                raise InsufficientMarchingMemory("Not enough free GPU memory for resident edge deduplication")
            keys = torch.empty(face_count * 3, device=device, dtype=torch.int64)
            face_offset = 0
            for start in range(0, tets.shape[0], chunk_size):
                tet_chunk = tets[start:start + chunk_size]
                n = tet_chunk.shape[0]
                counts = torch.empty(n, device=device, dtype=torch.int32)
                _count_surface_triangles[(triton.cdiv(n, 256),)](
                    tet_chunk, sdf, valids, counts, count_table, n, 256)
                offsets = counts.cumsum(0, dtype=torch.int32)
                n_faces = int(offsets[-1].item())
                if n_faces:
                    _emit_surface_edge_keys[(triton.cdiv(n, 256),)](
                        tet_chunk, sdf, counts, offsets, face_table, edges_table,
                        keys[3 * face_offset:], n, 256)
                    face_offset += n_faces
            unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            del keys
            if unique_keys.numel() >= 2**31:
                raise ValueError("mesh has too many unique edges for int32 faces")
            faces = inverse.to(torch.int32).reshape(-1, 3)
            del inverse
            ids = torch.stack((unique_keys >> 32, unique_keys & 0xFFFF_FFFF), dim=1).to(torch.int32)
            del unique_keys
    else:
        tets = tets.cpu().contiguous()
        ids, faces = _marching_tetrahedra_cpu_keys(
            tets, sdf, valids, count_table, face_table, edges_table, chunk_size, device
        )

    n_edges = ids.shape[0]
    end_points = torch.empty((n_edges, 2, 3), dtype=vertices.dtype, device=device)
    end_sdf = torch.empty((n_edges, 2, 1), dtype=sdf.dtype, device=device)
    end_scales = torch.empty((n_edges, 2, 1), dtype=scales.dtype, device=device)
    for start in range(0, n_edges, chunk_size):
        stop = min(start + chunk_size, n_edges)
        endpoint_ids = ids[start:stop].to(device=device, dtype=torch.long)
        end_points[start:stop] = vertices[endpoint_ids]
        end_sdf[start:stop, :, 0] = sdf[endpoint_ids]
        end_scales[start:stop, :, 0] = scales[endpoint_ids, 0]

    return (end_points, end_sdf), end_scales, faces, ids


def _marching_tetrahedra_cpu_keys(tets, sdf, valids, count_table, face_table, edges_table, chunk_size, device):
    with tempfile.TemporaryDirectory(prefix="marching_tetrahedra_") as temp_dir:
        key_path = os.path.join(temp_dir, "face_edge_keys.bin")
        face_count = 0
        with open(key_path, "wb") as key_file:
            for start in range(0, tets.shape[0], chunk_size):
                tet_chunk = tets[start:start + chunk_size].to(device)
                n = tet_chunk.shape[0]
                counts = torch.empty(n, device=device, dtype=torch.int32)
                _count_surface_triangles[(triton.cdiv(n, 256),)](
                    tet_chunk, sdf, valids, counts, count_table, n, 256)
                offsets = counts.cumsum(0, dtype=torch.int32)
                n_faces = int(offsets[-1].item())
                if n_faces:
                    keys = torch.empty(n_faces * 3, device=device, dtype=torch.int64)
                    _emit_surface_edge_keys[(triton.cdiv(n, 256),)](
                        tet_chunk, sdf, counts, offsets, face_table, edges_table,
                        keys, n, 256)
                    keys.cpu().numpy().tofile(key_file)
                    face_count += n_faces
                    del keys
                del tet_chunk, counts, offsets

        if face_count == 0:
            ids = torch.empty((0, 2), dtype=torch.int32)
            faces = torch.empty((0, 3), dtype=torch.int32)
        else:
            keys = np.memmap(key_path, dtype=np.int64, mode="r", shape=(face_count * 3,))
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            if len(unique_keys) >= 2**31:
                raise ValueError("mesh has too many unique edges for int32 faces")
            ids_np = np.empty((len(unique_keys), 2), dtype=np.int32)
            ids_np[:, 0] = unique_keys >> 32
            ids_np[:, 1] = unique_keys & 0xFFFF_FFFF
            ids = torch.from_numpy(ids_np)
            faces = torch.from_numpy(inverse.astype(np.int32, copy=False).reshape(-1, 3))
            del keys, unique_keys, inverse

    return ids, faces

