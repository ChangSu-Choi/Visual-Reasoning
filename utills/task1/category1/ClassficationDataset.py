from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


data_transforms = {
    "train": A.Compose([
        A.RandomResizedCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.RandomResizedCrop(224, 224),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


class ClassficationDataset(Dataset):
    
    def __init__(self, df, mode=None, transform=data_transforms):
        self.df = df
        self.mode = mode
        self.transforms = transform
        self.mode
        
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):

        try:
            sample = self.df.iloc[idx]
            ###
            target = sample["correct_answer_group_ID"][0] - 1 #  ori [1] or [2] so transe it for [0] or [1]

            q_img = sample["file_path"] + sample["Questions"][0]["images"][0]["image_url"] 
            a1_img = [sample["file_path"] + ans_img["image_url"] for ans_img in sample["answer1"][0]["images"]]
            a2_img = [sample["file_path"] + ans_img["image_url"] for ans_img in sample["answer2"][0]["images"]]
            read_qesimg_feature = cv2.imread(q_img)
            read_ansimg_feature1 = [cv2.imread(img) for img in a1_img]
            read_ansimg_feature2 = [cv2.imread(img) for img in a2_img]
            

            cvt_qimg_feature = cv2.cvtColor(read_qesimg_feature, cv2.COLOR_BGR2RGB)
            cvt_ansimg_feature1 = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in read_ansimg_feature1]
            cvt_ansimg_feature2 = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in read_ansimg_feature2]

            q_img_feature = self.transforms[self.mode](image=cvt_qimg_feature)["image"]
            a1_img_feature = [self.transforms[self.mode](image=img)["image"] for img in cvt_ansimg_feature1]
            a2_img_feature = [self.transforms[self.mode](image=img)["image"] for img in cvt_ansimg_feature2]

        except IOError:
            print(self.df.iloc[idx],"에서 문제 발생")
            pass
        
        return {
            "target": target,
            "q_img": q_img_feature,
            "a1_imgs": a1_img_feature,
            "a2_imgs": a2_img_feature,
            "file_path":sample["file_path"]
        }