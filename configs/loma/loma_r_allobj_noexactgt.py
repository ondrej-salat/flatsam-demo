from flatsam2.config import LoMaTrackerConfig, SamplingCfg

config = LoMaTrackerConfig(
    model="LoMaR",
    sampling=SamplingCfg(
        n_object=4096,
        total_budget=4096,
        nms_size=3,
        subpixel=True,
        subpixel_temp=0.5,
    ),
    exact_gt_mask_on_init=False,
    ransac_method="USAC_MAGSAC",
    ransac_threshold=5.0,
)
