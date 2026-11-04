from flatsam2.config import LoMaTrackerConfig, SamplingCfg

config = LoMaTrackerConfig(
    model="LoMaR",
    sampling=SamplingCfg(
        n_object=2000,
        total_budget=4096,
        nms_size=3,
        subpixel=True,
        subpixel_temp=0.5,
    ),
    init_mask_inflate_px=10,
    current_mask_inflate_px=10,
    ransac_method="USAC_MAGSAC",
    ransac_threshold=5.0,
)
