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
        sample = self.df.iloc[idx]
        target = sample["correct_answer_group_ID"][0]
        # file_paths = [sample["file_path"] + img for img in self.get_image_paths(sample)]


        img_paths = self.get_image_paths(sample)
        try:
            q1_img_features = [self.load_image(path) for path in img_paths[0][0]]
            q2_img_features = [self.load_image(path) for path in img_paths[0][1]]
            
            a1_img_features = [self.load_image(path) for path in img_paths[1][0]]
            a2_img_features = [self.load_image(path) for path in img_paths[1][1]]
            a3_img_features = [self.load_image(path) for path in img_paths[1][2]]

        except IOError:
            print(f"문제 발생 위치: {sample}")


        return {
            "target": target,
            "q1_imgs": q1_img_features,
            "q2_imgs": q2_img_features,
            "a1_imgs": a1_img_features,
            "a2_imgs": a2_img_features,
            "a3_imgs": a3_img_features,
            "file_path": sample["file_path"]
        }

    def get_image_paths(self, sample):
        # print('sample')
        # print(sample)
        return [
            [[sample["file_path"] + ans_img["image_url"] for ans_img in sample["question_img1"]["images"]],
            [sample["file_path"] + ans_img["image_url"] for ans_img in sample["question_img2"]["images"]]],
            
            [[sample["file_path"] + ans_img["image_url"] for ans_img in sample["answer_img1"]["images"]],
            [sample["file_path"] + ans_img["image_url"] for ans_img in sample["answer_img2"]["images"]],
            [sample["file_path"] + ans_img["image_url"] for ans_img in sample["answer_img3"]["images"]]]

        ]

    def load_image(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return self.transforms[self.mode](image=img)["image"]
    
