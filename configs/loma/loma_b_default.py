"""Default masked LoMa tracker configuration using LoMaB.

Output dir stem: loma_b_default
"""

from flatsam2.config import LoMaTrackerConfig, SamplingCfg

config = LoMaTrackerConfig(
    model="LoMaB",
    sampling=SamplingCfg(
        n_object=500,
        total_budget=4096,
        nms_size=3,
        subpixel=True,
        subpixel_temp=0.5,
    ),
    ransac_method="RANSAC",
    ransac_threshold=5.0,
)
