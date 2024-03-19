# Large Scale Visual Reasoning Dataset
📃[[PAPER]](https://ieeexplore.ieee.org/abstract/document/10067104)  
Author: **ChangSu Choi**, HyeonSeok Lim, Hayoung Jang, Juhan Park, Eunkyung Kim, KyungTae Lim  
[[MLP LAB]](https://sites.google.com/view/aailab), [[NIA]](https://www.nia.or.kr/site/nia_kor/main.do)그리고 [[EUCLIDSOFT]](https://www.euclidsoft.co.kr/)가 구축한 대규모 시각 추론 이미지 데이터셋.  


## Example  
![Question Images](./templates/question.png)

## 모델 실행 예시 코드   
### Dependency  
다음 명령어를 통해 필요한 패키지를 설치:
```
pip install -r requirements.txt
```
[[AI-Hub]](https://www.aihub.or.kr/)에서 접근 후 사용 가능  
[[카테고리 기반 추론 데이터]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71291), 
[[시각 상식 기반 추론 데이터]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71288), 
[[인과 관계 기반 추론 데이터]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71286), 
[[유사성 기반 추론 데이터]](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71279), 


## Traning(Classification_Train.py, SimCLR_Train.py, SupCLR_Train.py)  
사용 예시:
```
python Classification_Train.py \
    --train_link 'data/train_data' \
    --valid_link 'data/valid_data' \
    --task 1 \
    --category 2
```  
다음과 같이 하이퍼파라미터를 조정 가능합니다:
```
python Classification_Train.py \
    --input_dim 512 \
    --mlp_hidden 256 \
    --batch_size 128 \
    --model_name 'resnet50' \
    --mode 'train' \
    --num_workers 4 \
    --devices '0,1,2,3' \
    --epochs 10 \
    --clr_temperature 10 \
    --learning_rate 0.001 \
    --log_every_n_steps 100 \
    --model_save_path 'ckpt/save_dir' \
    --freeze \
    --resume_from_checkpoint 'ckpt/checkpoint_dir' \
    --train_link 'data/train_data' \
    --valid_link 'data/valid_data' \
    --task 1 \
    --category 2
```  

 
## Citation
Please cite the repo if you use the data or code in this repo.
```
@INPROCEEDINGS{10067104,
  author={Choi, ChangSu and Lim, HyeonSeok and Jang, Hayoung and Park, Juhan and Kim, Eunkyung and Lim, KyungTae},
  booktitle={2023 International Conference on Artificial Intelligence in Information and Communication (ICAIIC)}, 
  title={Semantic Similarity-based Visual Reasoning without Language Information}, 
  year={2023},
  volume={},
  number={},
  pages={107-111},
  keywords={Training;Deep learning;Visualization;Semantics;Training data;Transformers;Cognition;Visual Reasoning;Inference;Image similarity;Deep Learning},
  doi={10.1109/ICAIIC57133.2023.10067104}}

```
