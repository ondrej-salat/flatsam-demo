"""Shared RGB colors for paper figures."""

GT_RGB = (0, 190, 90)
FLATSAM_RGB = (220, 45, 120)
WOFTSAM_RGB = (30, 130, 230)
# Diagnostic-only: no paper caption names DINOv2 in colour, and this hue is
# close enough to SAM3_MASK_RGB that the two must never share a figure.
DINOV2_RGB = (225, 120, 30)
SAM3_MASK_RGB = (255, 100, 0)
# Pose implied by the mask branch alone; a hue no method uses, so it reads
# against both the green ground truth and the orange mask it is fitted to.
MASK_POSE_RGB = (130, 60, 230)


def rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB tuple to OpenCV's BGR order.

    Args:
        color: RGB color tuple.

    Returns:
        BGR color tuple.
    """
    red, green, blue = color
    return blue, green, red
