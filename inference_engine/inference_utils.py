import torch

from .utils.geometry import (
    register_camera_poses_kabsch_pytorch,
    apply_sim3_to_pose
)
from .utils.lsa import make_sp_graph, refine_depth_segments


def dict_to_device(data, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)

    return data


def sliding_window_t(arr, window_size, overlap, dim=0):
    step = window_size - overlap
    assert step > 0, "Overlap must be smaller than window_size"
    ndim = arr.ndim if hasattr(arr, "ndim") else arr.dim()
    assert 0 <= dim < ndim, f"Invalid dim {dim}, input has {ndim} dims"

    length = arr.shape[dim]
    windows = []

    for i in range(0, length, step):
        # Build slice for all dimensions
        slc = [slice(None)] * ndim
        slc[dim] = slice(i, i + window_size)
        window = arr[tuple(slc)]

        if i == 0 or window.shape[dim] > overlap:
            windows.append(window)
        else:
            break

    return windows


def sliding_window_l(li, window_size, overlap):
    step = window_size - overlap
    assert step > 0, "Overlap must be smaller than window_size"

    length = len(li)
    windows = []

    for i in range(0, length, step):
        # Slice the 1D list
        window = li[i: i + window_size]

        if i == 0 or len(window) > overlap:
            windows.append(window)
        else:
            break

    return windows


def aggregate_windows(windows, overlap, debug=False):
    if debug:
        return torch.cat(windows, dim=0)

    aggregated = [windows[0]]
    for i in range(1, len(windows)):
        aggregated.append(windows[i][overlap:])
    return torch.cat(aggregated, dim=0)


def estimate_pseudo_depth_and_intrinsics(local_points, eps=1e-8):
    N, H, W, _ = local_points.shape
    device = local_points.device

    c_x = W / 2.0
    c_y = H / 2.0
    u, v = torch.meshgrid(torch.arange(W, device=device),
                          torch.arange(H, device=device),
                          indexing='xy')
    u = u[None].expand(N, -1, -1)
    v = v[None].expand(N, -1, -1)

    x = local_points[..., 0]
    y = local_points[..., 1]
    z = local_points[..., 2]

    fx = torch.median((((u - c_x) * z) / (x + eps)).view(N, -1), dim=-1).values
    fy = torch.median((((v - c_y) * z) / (y + eps)).view(N, -1), dim=-1).values

    K_N = torch.zeros((N, 3, 3), dtype=torch.float32, device=device)
    K_N[:, 0, 0] = fx
    K_N[:, 1, 1] = fy
    K_N[:, 0, 2] = c_x
    K_N[:, 1, 2] = c_y
    K_N[:, 2, 2] = 1.0

    return z, K_N


def unproject_depth_to_local_points(depth, K):
    N, H, W = depth.shape
    device = depth.device

    # Create pixel grid [H, W]
    u = torch.arange(W, device=device).view(1, 1, W).expand(N, H, W)
    v = torch.arange(H, device=device).view(1, H, 1).expand(N, H, W)

    # If intrinsics are shared across batch, expand
    if K.ndim == 2:
        K = K.unsqueeze(0).expand(N, -1, -1)

    fx = K[:, 0, 0].view(N, 1, 1)
    fy = K[:, 1, 1].view(N, 1, 1)
    cx = K[:, 0, 2].view(N, 1, 1)
    cy = K[:, 1, 2].view(N, 1, 1)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    local_points = torch.stack((x, y, z), dim=-1)
    return local_points


def run_lsa_refinement(
        prev_local_points,
        cur_local_points,
        cur_conf,
        anchor_sp_graph,
        overlap,
        segmentation=None,
        diagnostics=None,
        intrinsic=None,
        window_id=None,
):
    """The single implementation of "segment the window, then correct its depth scales by LSA".

    Both `StreamingWindowEngine._registration_worker` and
    `StreamingWindowEngineLC._registration_worker` call this, and so does
    `register_extrinsic_windows`. Before IDEA-001 the same sequence of calls existed three times,
    which is why an observation point would have had to be written three times too.

    Returns `(corrected_local_points, tgt_sp_graph, scale_mask)` and leaves the decision of what to
    do with them to the caller: the canonical path multiplies immediately, the loop-closure path
    stores the mask for `aggregate_caches` because it defers every Sim(3) to aggregation.

    Observation points inserted here:

    * OP-1 / OP-2 — emitted inside `make_sp_graph` through the segmentation diagnostics.
    * OP-3 / OP-4 / OP-5 — emitted inside `align_adjacent_windows_depth_segments`.
    * OP-6 — the overlap depth-consistency measurement *before* and *after* the mask. This is the
      only genuinely new computation in the instrumentation, and it is the cheapest available
      falsification of "the gain travels through LSA": if the correction does not make the two
      windows agree on the shared frames, any ATE improvement comes from elsewhere.

    With `diagnostics=None` (the default) no extra work is done and the returned tensors are
    identical to the pre-IDEA-001 code path.
    """
    from .utils.lsa import segmentation_config_or_default

    segmentation = segmentation_config_or_default(segmentation)

    irls_kwargs = {
        "iters": segmentation.irls.iters,
        "eps": segmentation.irls.eps,
        "stop_tol": segmentation.irls.stop_tol,
        "clamp_min": segmentation.irls.clamp_min,
    }

    tgt_pcd = cur_local_points.cpu().numpy()
    tgt_conf = cur_conf.cpu().numpy() if cur_conf is not None else None
    if tgt_conf is not None:
        # The Baseline sees only the OVERLAP frames' confidence. That slice is what fixes each
        # frame's confidence cut, hence each frame's high-confidence depth range, hence each
        # frame's merge threshold. Passing the whole window instead computes one threshold from
        # the window's full depth range, which can be an order of magnitude larger for a near
        # frame that shares a window with distant geometry.
        tgt_conf = tgt_conf[:overlap]

    tgt_sp_graph = make_sp_graph(
        tgt_pcd[..., -1],
        conf_map=tgt_conf,
        point_map=tgt_pcd,
        intrinsic=None if intrinsic is None else (
            intrinsic.cpu().numpy() if hasattr(intrinsic, "cpu") else np.asarray(intrinsic)
        ),
        segmentation=segmentation,
        diagnostics=diagnostics,
    )

    before = None
    if diagnostics is not None:
        from .segmentation_diagnostics import overlap_depth_consistency

        before = overlap_depth_consistency(
            source_points=prev_local_points[-overlap:].cpu().numpy(),
            target_points=tgt_pcd[:overlap],
            stride=4,
        )

    scale_mask = refine_depth_segments(
        prev_local_points.cpu().numpy(),
        tgt_pcd,
        anchor_sp_graph,
        tgt_sp_graph,
        overlap,
        corr_iou_thresh=segmentation.corr_iou_thresh_inter,
        irls=irls_kwargs,
        diagnostics=diagnostics,
    )

    if diagnostics is not None:
        from .segmentation_diagnostics import (
            OP_OVERLAP_CONSISTENCY,
            mask_consistency_delta,
            overlap_depth_consistency,
        )
        from .utils.lsa import record_scale_mask

        after = overlap_depth_consistency(
            source_points=prev_local_points[-overlap:].cpu().numpy(),
            target_points=(cur_local_points * scale_mask.to(cur_local_points.device))
            .cpu().numpy()[:overlap],
            stride=4,
        )
        diagnostics.record_stage(
            OP_OVERLAP_CONSISTENCY,
            mask_consistency_delta(before, after),
        )
        record_scale_mask(scale_mask.cpu().numpy(), diagnostics)

    return cur_local_points * scale_mask.to(cur_local_points.device), tgt_sp_graph, scale_mask


def register_extrinsic_windows(
        pcd_windows,
        cam_windows,
        mask_windows,
        overlap,
        depth_refine=False,
        debug=False
):
    camera_poses = [cam_windows[0]]
    local_points = [pcd_windows[0]]

    _, intrinsic_ = estimate_pseudo_depth_and_intrinsics(pcd_windows[0])
    ref_intrinsic = intrinsic_[0]
    pcd_windows = [unproject_depth_to_local_points(pcd[..., -1], ref_intrinsic) for pcd in pcd_windows]

    anchor_cam_pose = cam_windows[0]
    anchor_pcd = pcd_windows[0]
    anchor_sp_graph = None
    if depth_refine:
        anchor_sp_graph = make_sp_graph(
            pcd_windows[0][..., -1].cpu().numpy(),
            mask_windows[0].cpu().numpy()
        )

    for i in range(1, len(cam_windows)):
        valid_mask = mask_windows[i - 1][-overlap:] & mask_windows[i][:overlap]
        tgt_pcd = pcd_windows[i]

        tgt_cam_pose = cam_windows[i]
        s_d, R, t = register_adjacent_windows(
            anchor_pcd[-overlap:],
            tgt_pcd[:overlap],
            anchor_cam_pose[-overlap:],
            tgt_cam_pose[:overlap],
            valid_mask
        )

        # anchor_cam_pose = tgt2src_sim3 @ tgt_cam_pose
        anchor_cam_pose = apply_sim3_to_pose(tgt_cam_pose, s_d, R, t)
        tgt_pcd = s_d * tgt_pcd

        # full numpy processing pipeline
        if depth_refine:
            tgt_pcd = tgt_pcd.cpu().numpy()
            tgt_sp_graph = make_sp_graph(
                tgt_pcd[..., -1],
                mask_windows[i].cpu().numpy()
            )

            tgt_pcd_scaled = refine_depth_segments(
                anchor_pcd.cpu().numpy(),
                tgt_pcd,
                anchor_sp_graph,
                tgt_sp_graph,
                overlap
            ).to(anchor_pcd.device)

            anchor_pcd = tgt_pcd_scaled
            anchor_sp_graph = tgt_sp_graph
        else:
            anchor_pcd = tgt_pcd

        if debug:
            local_points.append(anchor_pcd)
            camera_poses.append(anchor_cam_pose)
        else:
            local_points.append(anchor_pcd[overlap:])
            camera_poses.append(anchor_cam_pose[overlap:])

    return torch.cat(camera_poses, dim=0), torch.cat(local_points, dim=0)


def register_adjacent_windows(
        src_pcd_overlap,  # N, H, W, 3
        tgt_pcd_overlap,  # N, H, W, 3
        src_cam_overlap,  # N, 4, 4
        tgt_cam_overlap,  # N, 4, 4
        mask,  # N, H, W
        register_func=register_camera_poses_kabsch_pytorch
):
    s_d = align_cam_pts_irls(
        tgt_pcd_overlap,
        src_pcd_overlap,
        mask
    )

    R, t = register_func(tgt_cam_overlap, src_cam_overlap, scale=s_d)
    return s_d, R, t


def align_cam_pts_irls(
        src_pts,  # (N,H,W,3)
        tgt_pts,  # (N,H,W,3)
        mask=None,  # (N,H,W) optional boolean
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6
):
    if mask is not None:
        src_pts = src_pts[mask]
        tgt_pts = tgt_pts[mask]

    src_pts = src_pts.reshape(-1, 3)
    tgt_pts = tgt_pts.reshape(-1, 3)
    num = (src_pts * tgt_pts).sum()
    den = (src_pts ** 2).sum()
    s_d = torch.clamp(num / den, min=clamp_min)

    for _ in range(iters):
        # Compute residuals between scaled predictions and ground truth
        d_res = s_d * src_pts - tgt_pts
        res = torch.sqrt((d_res ** 2).sum(-1)) + eps
        w = 1.0 / res

        # Update 's' using weighted sums
        num = (w[:, None] * src_pts * tgt_pts).sum()
        den = (w[:, None] * src_pts ** 2).sum()
        s_d_new = torch.clamp(num / den, min=clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break  # early stopping

    return s_d
