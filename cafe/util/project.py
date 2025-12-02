import numpy as np


def project_to_segments(
    x: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
):
    """project points to segments

    Args:
        x (np.ndarray): point array (n_obs, n_dim)
        segment_start (np.ndarray): segement start array (n_segement, n_dim)
        segment_end (np.ndarray): segement start array (n_segement, n_dim)

    Raises:
        ValueError: The number of columns of 'x', 'segment_start' and 'segment_end' should be exactly the same
        ValueError: The number of rows of 'segment_start' and 'segment_end' should be exactly the same

    Returns:
        dict: dict with key-value pair :
            "x_proj": projection points of raw points on segments (n_obs, n_dim)
            "distance": distance of raw proints to projection points (n_obs)
            "segment": segement index of projection points(n_obs)
            "progression": progression of projection points on segement (n_obs)
    """

    # Ensure inputs are numpy arrays
    x = np.array(x)
    segment_start = np.array(segment_start)
    segment_end = np.array(segment_end)

    # Check dimensions
    if x.shape[1] != segment_start.shape[1] or x.shape[1] != segment_end.shape[1]:
        raise ValueError("The number of columns of 'x', 'segment_start' and 'segment_end' should be exactly the same")
    if segment_start.shape[0] != segment_end.shape[0]:
        raise ValueError("The number of rows of 'segment_start' and 'segment_end' should be exactly the same")

    # Determine dimensionalities
    ncols = x.shape[1]
    npts = x.shape[0]
    nsegs = segment_start.shape[0]

    # Calculate lengths of the segments
    diff = segment_end - segment_start
    length = np.sum(diff**2, axis=1)

    # Initialize output objects
    x_proj = np.zeros((npts, ncols))
    distance = np.zeros(npts)
    segment = np.zeros(npts, dtype=int)
    progression = np.zeros(npts)

    # Iterate over points in x
    for i in range(npts):
        p = x[i, :]

        # Store information on the closest segment
        best_segment = -1
        best_progression = -1
        best_distance = np.inf

        # Iterate over the segments
        for segi in range(nsegs):
            # If test_progression is between 0 and 1, the projection is on the segment.
            # Else it is outside the segment, <0 before the start, >1 after the end
            test_progression = np.dot(diff[segi, :], p - segment_start[segi, :]) / length[segi]

            # Clamp progression between 0 and 1,
            # If the projection is on the segment, projection is valid
            # Else move the projection to the closest endpoint
            test_progression = max(0.0, min(1.0, test_progression))

            # Calculate position of projection and the distance
            test_p_proj = segment_start[segi, :] + test_progression * diff[segi, :]
            test_distance = np.sum((test_p_proj - p) ** 2)

            # If this is better than what was found earlier, store it
            if test_distance < best_distance:
                best_distance = test_distance
                best_segment = segi
                best_progression = test_progression
                x_proj[i, :] = test_p_proj

        distance[i] = best_distance
        segment[i] = best_segment + 1  # Increase by 1 for R-like indexing
        progression[i] = best_progression

    out = {"x_proj": x_proj, "distance": distance, "segment": segment, "progression": progression}

    return out


def project_to_divergence_regions(
    x: np.ndarray,
):
    # project points to divergence regions

    x_proj = None
    distance = None
    segment = None
    progression = None

    out = {"x_proj": x_proj, "distance": distance, "segment": segment, "progression": progression}

    return out
