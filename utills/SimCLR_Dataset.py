from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2




data_transforms = {
    "train": A.Compose([
        A.RandomResizedCrop(224, 224),
        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8),
        A.ToGray(p=0.2),    
        A.Normalize(),
        ToTensorV2(p=1.0)], p=1.0),
    
    "valid": A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(p=1.0)], p=1.0)
}


class SimCLR_Dataset(Dataset):
    
    def __init__(self, df, mode=None, transform=data_transforms):
        self.df = df
        self.mode = mode
        self.transform = transform
        self.mode
        
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        
        img = sample["image_path"]

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        if self.transform: #이미지 데이터의 크기 및 각도등을 변경
            img1 = self.transform[self.mode](image=img)["image"]
            img2 = self.transform[self.mode](image=img)["image"]
            
        if img1.shape != (3, 224, 224) or img2.shape != (3, 224, 224):
            print(f"Warning: Unexpected tensor size for image {sample['image_path']}. img1 shape: {img1.shape}, img2 shape: {img2.shape}")

        return {
            "img_feature_1": img1,
            "img_feature_2": img2,
        }
        
