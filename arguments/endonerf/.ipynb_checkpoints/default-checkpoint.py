ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    position_lr_max_steps = 7000,
    # single_view_weight_from_iter = 700,
    percent_dense = 0.01,
    opacity_reset_interval = 10000,
    # prune_interval = 10000,
    w_coefs_lambda = 0.0,

    
    wo_image_weight = True,
    single_view_weight = 0.005,
    regularize_geometry_only_mask = True,
    single_view_weight_from_iter = 2500,

    
    scale_loss_weight = 100,

    tv_depth_loss_weight = 0.015,
    tv_depth_loss_from_iter = 700000,

    tv_color_loss_weight = 0.010,
    tv_color_loss_from_iter = 1200000,

    clean_noise = False,
    align_loss_from_iter = 700000,
    align_loss_weight = 0.02,
    # feature_lr = 0.005,
    # opacity_lr = 0.075,
    # scaling_lr = 0.01,
    # rotation_lr = 0.0025,

    num_frame = 260 ,
    frame_segmented = 260,



    color_weight = 1,
    depth_weight = 1,
    sharp_metric_stop_iter = 0,
    adpt_weight = False,

    # adpt_sampling = False,
    # adpt_weight = False,
    fft_weight_l = 0.000325,
    fft_weight_h = 0.00025,
    fft_from_iter = 0,
    fft_D0 = 0.5,
    fft_D = 1.0,
    T0 = 1000,
    T = 3000,

)

ModelHiddenParams = dict(
    curve_num = 17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )
