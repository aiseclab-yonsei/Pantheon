from PIL import Image
import numpy as np

import torchvision
from torch.utils.data import Dataset

from network.network_modules import PairGenerator
from dataloaders.get_df import get_df_v1

import warnings
warnings.simplefilter("ignore", UserWarning)


# class Train(Dataset):
#     def __init__(self, cfg):
#         self.image_size = cfg.image_size

#         self.image_dir, self.df = get_df_v1(cfg, is_train=True)
#         self.pg = PairGenerator(tau=cfg.tau)

#         self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                           torchvision.transforms.Resize(self.image_size),
#                                                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                                                                            std=(0.229, 0.224, 0.225)),
#                                                           ])


#     def __getitem__(self, idx):

#         sample = dict()

#         for im_num in range(self.im_num):
#             img_name = self.image_dir + self.df['image_name'].iloc[self.idx[im_num][idx]]
#             img_mos = self.df['MOS'].iloc[self.idx[im_num][idx]]

#             img = Image.open(img_name).convert("RGB")
#             img_trans = self.transforms(img)

#             sample[f'img_{im_num}_mos'] = img_mos
#             sample[f'img_{im_num}_group'] = self.group[im_num][idx]
#             sample[f'img_{im_num}'] = img_trans

#         return sample

#     def __len__(self):
#         return len(self.idx.transpose(1, 0))

#     def get_pair_lists(self, batch_size, batch_list_len, im_num, uniform_select=False):
#         self.im_num = im_num
#         self.idx, self.group = self.pg.get_train_im_list(mos=np.round_(self.df['MOS'].values, 0), batch_size=batch_size, batch_list_len=batch_list_len, im_num=self.im_num, uniform_select=uniform_select)


class Test(Dataset):
    def __init__(self, cfg):
        self.image_size = cfg.image_size
        self.df_test = get_df_v1(cfg, is_train=False)

        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Resize(self.image_size),
                                                          torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                                           std=(0.229, 0.224, 0.225)),
                                                          ])

    def __getitem__(self, idx):
        sample = dict()

        # img_name = self.image_dir + self.df_test['image_name'].iloc[idx]
        img_name = self.df_test['image_name'].iloc[idx]

        # img_mos = self.df_test['MOS'].iloc[idx]
        try:
            img = Image.open(img_name).convert("RGB")
            img_trans = self.transforms(img)
        except Exception as e:
            print(f"\n[❌ ERROR] 이미지 로딩 중 오류 발생!")
            print(f"🖼️ 이미지 경로: {img_name}")
            print(f"📌 에러 메시지: {e}")
            # raise RuntimeError(f"💥 이미지 로딩 실패: {img_name}")  # 실행을 강제로 멈춤
            print(f"➡️ 이미지 대신 NaN 텐서를 반환합니다.\n")

            # transforms가 입력으로 받는 dummy 이미지 생성
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))  # 크기는 모델 입력에 맞게
            img_trans = self.transforms(dummy).float() * float('nan')

        sample['img_path'] = img_name
        # sample['img_mos'] = img_mos
        sample[f'img'] = img_trans

        return sample

    def __len__(self):
        return len(self.df_test)

# class Ref(Dataset):
#     def __init__(self, cfg):
#         self.image_size = cfg.image_size
#         self.image_dir, self.df_base, _, _ = get_df_v1(cfg, is_train=False)

#         self.pg = PairGenerator(tau=cfg.tau)
#         self.get_pair_lists(cfg.batch_size-1)

#         self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                           torchvision.transforms.Resize(self.image_size),
#                                                           torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                                                                            std=(0.229, 0.224, 0.225)),
#                                                           ])

#     def __getitem__(self, idx):
#         sample = dict()

#         img_name = self.image_dir + self.df_base['image_name'].iloc[self.idx_0[idx]]
#         img_mos = self.df_base['MOS'].iloc[self.idx_0[idx]]
#         img = Image.open(img_name).convert("RGB")
#         img_trans = self.transforms(img)

#         sample['img_path'] = img_name
#         sample['img_mos'] = img_mos
#         sample[f'img'] = img_trans

#         return sample

#     def __len__(self):
#         return len(self.idx_0)

#     def get_pair_lists(self, batch_size):
#         self.idx_0, self.group_0 = self.pg.get_test_im_list(mos=np.round_(self.df_base['MOS'].values, 0), batch_size=batch_size, random_choice=True)
