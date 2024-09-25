import torch
import torch.nn as nn
import torch.nn.functional as F

class FGAM(nn.Module):
    def __init__(self, n_classes, dim_time_varying, dim_static, n_embedding, n_hid, batch_norm=False, reweighting_weights=None):
        # FGAM 클래스 초기화 함수
        super(FGAM, self).__init__()

        # 입력된 변수들 초기화
        self.n_classes = n_classes  # 출력 클래스 수
        self.dim_time_varying = dim_time_varying  # 시간 가변 특성 차원
        self.dim_static = dim_static  # 고정 특성 차원
        self.n_embedding = n_embedding  # 임베딩 차원
        self.n_hid = n_hid  # 숨겨진 계층 수
        self.batch_norm = batch_norm  # 배치 정규화 사용 여부
        self.wt = reweighting_weights

        # 임베딩 계층 리스트 초기화 (시간 가변 특성 개수만큼)
        self.embedding_layers = nn.ModuleList()

        # 시간 가변 특성마다 별도의 임베딩 네트워크 생성
        for i in range(self.dim_time_varying):
            branch = nn.ModuleList()
            # 첫 번째 계층은 1차원 입력을 n_embedding 크기로 변환
            branch.append(nn.Linear(1, self.n_embedding))
            for _ in range(n_hid):
                # 숨겨진 계층들을 추가
                branch.append(nn.Linear(self.n_embedding, self.n_embedding))
                if self.batch_norm:
                    branch.append(nn.BatchNorm1d(self.n_embedding))  # 배치 정규화가 있을 경우 추가
            # 마지막은 128 차원으로 변환한 후 1차원 출력
            branch.append(nn.Linear(self.n_embedding, 128))
            branch.append(nn.Linear(128, 1))
            self.embedding_layers.append(branch)  # 각 시간 가변 특성에 대해 네트워크를 리스트에 추가
        self.linear = nn.Linear(self.dim_time_varying, self.n_classes)  # 최종 출력 계층

        # 고정 특성에 대해 가중치를 계산하기 위한 네트워크 (logits0을 위한 가중치와 bias 계산)
        self.weights_module = nn.ModuleList()
        self.weights_module.append(nn.Linear(self.dim_static, 128))
        self.weights_module.append(nn.Linear(128, 128))
        self.weight = nn.Linear(128, self.dim_time_varying * 1)  # 가중치 계산
        self.bias = nn.Linear(128, 1)  # bias 계산

        # logits1에 대한 가중치와 bias 계산을 위한 또 다른 네트워크
        self.weights_module2 = nn.ModuleList()
        self.weights_module2.append(nn.Linear(self.dim_static, 128))
        self.weights_module2.append(nn.Linear(128, 128))
        self.weight2 = nn.Linear(128, self.dim_time_varying * 1)  # 가중치 계산. 각 시간 가변 특성마다 고정 특성에 따른 가중치를 계산(ex. 나이에 따른 CK의 영향 정도)
        self.bias2 = nn.Linear(128, 1)  # bias 계산

    def forward(self, static, time_varying):
        # FGAM의 순전파 (forward) 함수

        res = []
        # 각 시간 가변 특성에 대해 처리
        for i in range(self.dim_time_varying):
            x = time_varying[i]  # 각 시간 가변 특성을 꺼냄
            for j, op in enumerate(self.embedding_layers[i]):
                # 배치 정규화 앞에는 ReLU를 적용하지 않음
                if (j < (len(self.embedding_layers[i])-1) and
                        (self.embedding_layers[i][j+1]._get_name() ==
                         'BatchNorm1d')) or\
                        (j == (len(self.embedding_layers[i])-1)):
                    x = op(x)  # 배치 정규화가 있거나 마지막 계층일 경우 ReLU 없이 연산
                else:
                    x = F.relu(op(x))  # 그 외에는 ReLU 활성화 함수 적용
            res.append(x)  # 결과 리스트에 저장

        x = torch.cat(res, 1)  # 시간 가변 특성들을 결합 (행렬 차원으로 이어붙임)

        # 고정 특성(static)으로 가중치 및 bias 계산
        w = static
        for i, op in enumerate(self.weights_module):
            w = F.relu(op(w))  # 네트워크 통과 시 ReLU 적용
        weight = self.weight(w)  # 가중치 계산
        bias = self.bias(w)  # bias 계산
        logits0 = torch.sum(x * weight, dim=1, keepdim=True) + bias  # logits0 계산

        # 두 번째 logits 계산 (logits1)
        w2 = static
        for i, op in enumerate(self.weights_module2):
            w2 = F.relu(op(w2))
        weight2 = self.weight2(w2)  # 가중치 계산
        bias2 = self.bias2(w2)  # bias 계산
        logits1 = torch.sum(x * weight2, dim=1, keepdim=True) + bias2  # logits1 계산

        logits = torch.cat((logits0, logits1), 1)  # logits0과 logits1을 결합하여 최종 출력
        
        # Incorporate the reweighting weights, if provided
        if self.wt is not None:
            reweighting_weights = torch.tensor(self.wt[['weight']], dtype=torch.float32)
            logits = logits * reweighting_weights.unsqueeze(1)  # Ensure broadcasting works
            
        return logits