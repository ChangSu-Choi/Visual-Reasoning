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
        A.Resize(224, 224),
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

        sample = self.df.iloc[idx]
        if sample["is_correct"]:
            target = 1
        else:
            target = 0
        
        q_img = [sample["file_path"] + ans_img["image_url"] for ans_img in sample["question"][0]["images"]] 
        q1_img = cv2.imread(q_img[0])
        q2_img = cv2.imread(q_img[1])
        q3_img = cv2.imread(q_img[2])


        q1_img_feature = cv2.cvtColor(q1_img, cv2.COLOR_BGR2RGB)
        q2_img_feature = cv2.cvtColor(q2_img, cv2.COLOR_BGR2RGB)
        q3_img_feature = cv2.cvtColor(q3_img, cv2.COLOR_BGR2RGB)

        q1_img_feature = self.transforms[self.mode](image=q1_img_feature)["image"]
        q2_img_feature = self.transforms[self.mode](image=q2_img_feature)["image"]
        q3_img_feature = self.transforms[self.mode](image=q3_img_feature)["image"]

        a_img = sample["file_path"] + sample["answer"][0]["images"][0]["image_url"]
        a_img_feature = cv2.imread(a_img)
        a_img_feature = cv2.cvtColor(a_img_feature, cv2.COLOR_BGR2RGB)
        a_img_feature = self.transforms[self.mode](image = a_img_feature)["image"]

        # except Exception as e:
        #     print(e)
        #     print(self.df.iloc[idx],"에서 문제 발생")
        #     pass
        
        return {
            "target": target,
            "q1_imgs": q1_img_feature,
            "q2_imgs": q2_img_feature,
            "q3_imgs": q3_img_feature,
            "a_img": a_img_feature
        }