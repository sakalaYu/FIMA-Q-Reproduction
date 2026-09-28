class Config:
    def __init__(self):
        # Same W4A4 calibration and QDrop settings as configs/4bit/best.py.
        self.optim_size = 1024
        self.calib_size = 128
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        self.w_bit = 4
        self.a_bit = 4
        self.qconv_a_bit = 8
        self.qhead_a_bit = 4
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        self.keep_gpu = True

        # Fixed forward-secant reconstruction settings.
        self.optim_metric = 'forward_secant'
        self.temp = 20
        self.k = 5
        self.p1 = 1.0
        self.p2 = 1.0
        self.dis_mode = 'q'
        self.recon_iters = 20000

        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5
