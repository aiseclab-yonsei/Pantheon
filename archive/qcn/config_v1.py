import os
from utils.util import get_current_time, write_log

class ConfigV1:
    def __init__(self):
        # dataset
        # self.dataset_name = 'SPAQ'
        #Dataset_name: UADFV

        ##폴더 명 ## 'FaceForensics++' 'UADFV' 'Celeb-DF-v1' 'Celeb-DF-v2' 
        ##### ['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures' ]   


        self.dataset_name ='AIGIBench_InsightFace_q_90'  # 'sd1.4_q60' 'sd1.4_q90' 'sd1.4_q100' 'sd1.4_q120' 'sd1.4_q140' 'sd1.4_q160' 'sd1.4_q180' 'sd1.4_q200'
        # self.dataset_root = f'/home/compu/eunbi/Datasets/{self.dataset_name}'
        self.dataset_root = f'/srv/shared/dataset/origin/wilddeepfake_origin/wilddeepfake_real'
        # self.datasplit_root = f'/home/compu/eunbi/iqa/0.module/preprocessing/{self.dataset_name}/'
        self.datasplit_root = f'/home/compu/eunbi/iqa/0.module/preprocessing/{self.dataset_name}/'

        # self.datasplit_root = f'/home/ysu/eunbi/iqa/qcn/datasplit/visualization/'
        # self.datasplit_root = f'/home/ysu/eunbi/iqa/qcn/datasplit/{self.dataset_name}/original_sequences/c23/'
        self.training_scheme = 'random_split' # for KonIQ10K dataset
        self.split = 1

        # network
        self.backbone = 'resnet50'
        self.model_name = 'CTV29'

        # Score pivots
        self.spv_num = 101

        self.reference_point_num=101
        self.score_pivot_score = list(range(self.spv_num))  # ⬅️ 이 줄 추가!


        # exp description
        self.exp_name = ''

        # encoder
        self.image_size = 384
        self.reduced_dim = 256

        # comparison transformer
        self.dec_layers = [1] * 3

        # train & test
        self.batch_size = 18
        self.test_batch_size = 10
        self.num_list = 9
        self.start_iter = 0
        self.start_eval = 60
        self.eval_freq = 5

        self.im_list_len = 150
        self.im_num = 2

        # hyperparameters
        self.tau = 0
        self.margin = 0.1

        # optimization
        self.optim = 'AdamW'
        self.scheduler = 'cosinewarmup'
        self.lr = 1e-5 * 5
        self.weight_decay = 1e-4 * 5

        self.epoch = 100
        self.test_first = True

        # misc
        self.num_workers = 0
        self.gpu = '0,1'
        self.wandb = False

        # logging
        self.save_folder_parent = './results/'
        self.save_folder = f'{self.save_folder_parent}/{self.dataset_name}/{self.model_name}/Back_{self.backbone}_M_{self.model_name}_C{self.reduced_dim}_T{self.tau:.2f}_{self.exp_name}_{get_current_time()}'
        # os.makedi/rs(self.save_folder, exist_ok=True)

        # self.load = False
        #### 기존 학습된 모델 가져오기 ####
        self.load = True

        if self.load:
            self.ckpt_file = 'SRCC_Epoch_94_SRCC_0.9246_PCC_0.9265_MAE_6.1870.pth'
            self.init_model = f'/home/compu/eunbi/iqa/qcn/ckpt/SPAQ/Split_1/{self.ckpt_file}'
        #self.log_file = self.log_configs()

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_folder}/{log_file}'):
            log_file = open(f'{self.save_folder}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_folder}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        # log_file.close()
        return log_file


if __name__ == "__main__":
    c = ConfigV1()
    print('debug... ')


