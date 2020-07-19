# 파이썬 모듈
import numpy as np
import csv

np.random.seed(1234)

# 하이퍼 파라미터 정의
RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001


# 메인함수 정의
def regression_1_layer_main(epoch_count=10, mb_size=10, report=1):

    load_abalone_dataset()  # 데이터 불러오기

    init_model()  # 모델 초기화

    train_and_test(epoch_count, mb_size, report)  # 학습 및 테스트 수행


def load_abalone_dataset():
    with open('abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # 다음 행부터 가져오기(스킵)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_count, output_count  # 데이터의 입출력 벡터 정보 저장
    input_count, output_count = 10, 1       # 입출력 벡터의 크기 지정(10개의 입력 속성, 1개의 출력)
    # 입력 벡터 = 10
    # 출력 벡터 = 1
    # 지정해준 크기만큼 0값의 행렬 생성!
    data = np.zeros([len(rows), input_count + output_count])

    # 원-핫 벡터 처리 [I - 첫번째 열, M - 두번째 열, F - 세번째 열]
    for n, row in enumerate(rows):
        if row[0] == 'I':
            data[n, 0] = 1
        if row[0] == 'M':
            data[n, 1] = 1
        if row[0] == 'F':
            data[n, 2] = 1
        data[n, 3:] = row[1:]       # 원핫인코팅 후 나머지 7개의 속성들(따로 인코딩할 필요 없는) 복사


# 파라미터 초기화 함수 정의
def init_model():

    global weight, bias, input_count, output_count

    weight = np.random.normal(
        RND_MEAN, RND_STD, [input_count, output_count])       # 랜덤한 가중치 생성 10*1 행렬 [] .shape 결정

    # 1*1 편향 -> 단층이기 떄문에, 0으로 가득한 array 생성 (초기 편향)
    bias = np.zeros([output_count])


# 학습 및 평가 함수 정의
def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()        # 데이터 가져오기

    for epoch in range(epoch_count):        # 에폭 반복 수행(학습을 몇회 진행할 것인가?)
        losses, accs = [], []       # 한 에폭마다의 손실과 정확도 기록

        for nth in range(step_count):     # 미니배치 처리된 횟수 만큼 반복 수행
            # 학습하는 데이터의 크기, 횟수(한 학습과정이 몇번의 단계를 거칠 것인가?)
            train_x, train_y = get_train_data(mb_size, nth)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            acc_dp = round(np.mean(accs)/acc, 5)
            print(
                f'Epoch {epoch+1}: loss={np.mean(losses)}, accuracy={acc_dp}')

    final_acc = run_test(test_x, test_y)
    final_acc = round(final_acc, 5)
    print(f'\nFinal Test: final accuracy = {final_acc}')


# 학습 및 평가 데이터 획득 함수 정의
def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    # data.shape = (n,10) -> np.arange(n) -> array[0,1,2,....,n-1]
    shuffle_map = np.arange(data.shape[0])
    # shuffle_map의 역할: 데이터를 섞어 같은 데이터로 훈련하지 않도록 data의 인덱스를 담은 배열
    # shuffle_map의 원소는 data의 인덱스임 -> shuffle_map을 섞어 원본 data를 훼손하지 않고 랜덤하게 추출
    np.random.shuffle(shuffle_map)      # 무작위로 데이터 섞기(야바위)

    # 전체 데이터의 80%를 훈련 데이터로 사용
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count       # 한번의 에폭에서 학습하는 횟수


def get_test_data():
    global data, shuffle_map, test_begin_idx, output_count
    test_data = data[shuffle_map[test_begin_idx:]]
    # 테스트 데이터의 종속, 독립 변수 분할
    return test_data[:, :-output_count], test_data[:, -output_count:]


def get_train_data(mb_size, nth):       # 미니배치 크리, 미니배치 실행 순서
    global data, shuffle_map, test_begin_idx, output_count
    if nth == 0:    # 데이터 섞기 (각 epoch의 처음에 한번 섞음)
        np.random.shuffle(shuffle_map[:test_begin_idx])
        # 훈련이 끝난 모델이 test_data를 이용해 예측한 값과 실제 값을 확인해 정확도와 오차율을 계산하기 위해 따로 분리해둠!!
    # 만든 shuffle_map 인덱스로 배열(array) 인덱싱
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_count], train_data[:, -output_count:]


# 학습 및 평가 데이터 획득 함수 정의
def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accurancy = eval_accuracy(output, y)
    # 데이터를 통해 손실과 정확도 파악
    # 항상 순전파의 역순으로 수행
    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)
    # 산출된 손실과 정확도를 통해 역전파를 실행
    return loss, accurancy


# 학습 및 평가 데이터 획득 함수 정의
def run_test(x, y):  # 정확도 판단
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


# 단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
def forward_neuralnet(x):       # 순전파 (단순 계산)
    global weight, bias
    # 편향이 더해진 입력 벡터와 가중치 벡터에 대한 기본적인 신경망 연산
    output = np.matmul(x, weight) + bias        # Y = W*X + b
    return output, x        # x : 역전파에 필요한 보조정보로 활용


def backprop_neuralnet(G_output, x):        # 역전파(갱신)
    global weight, bias
    g_output_w = x.transpose()      # 입력벡터 x에 대한 전치(n*10 -> 10*n)

    # 가중치 조절값 10*n 행렬과 G_output(n*1행렬) -> 10*1 행렬
    G_w = np.matmul(g_output_w, G_output)

    G_b = np.sum(G_output, axis=0)      # 편향 조절값

    weight -= LEARNING_RATE * G_w       # 가중치 갱신
    bias -= LEARNING_RATE * G_b     # 편향 갱신


# 후처리 과정에 대한 순전파 및 역전파 함수 정의
def forward_postproc(output, y):        # 퍼셉트론의 손실확인
    diff = output - y       # 오차 n*1
    square = np.square(diff)        # 오차 제곱 n*1 행렬
    loss = np.mean(square)      # 평균 오차 제곱(MSE)
    return loss, diff


def backprop_postproc(G_loss, diff):        # 역전파 갱신 후 처리
    shape = diff.shape      # (n, 10)

    # n*10의 행렬 / product 배열간의 원소 곱(=scalsr n*10)
    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))  # np.abs((output-y)/y) = 오차
    return 1 - mdiff


regression_1_layer_main()


# 파라미터 확인
print(weight)
print(bias)


# 새로운 입력 벡터 X에 대한 예측
# 훈련된 모델을 통한 결과 예측
x = np.array([0, 1, 0, 0.44, 0.3, 0.08, 0.5, 0.23, 0.11, 0.2])
output = forward_neuralnet(x)
print(output)


# 하이퍼파라미터 수정해가며 실험
LEARNING_RATE = 0.1
regression_1_layer_main(epoch_count=100, mb_size=100, report=20)
