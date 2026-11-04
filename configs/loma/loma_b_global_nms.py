"""LoMaB config: global NMS keypoint sampling.

Up to 500 object keypoints; remaining budget (total 4096) filled from background.

Output dir stem: loma_b_global_nms
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
