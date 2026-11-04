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
    init_mask_inflate_px=15,
    current_mask_inflate_px=15,
    ransac_method="USAC_MAGSAC",
    ransac_threshold=5.0,
)
