#####################################################################
# Word2Vec 모델을 간단하게 구현해봅니다.

import numpy as np
import tensorflow as tf

import tool as tool

# data loading
data_path = './sample.csv'
title, contents = tool.loading_data(data_path, eng=False, num=False, punc=False)  # 트레이닝 data read
test_title, test_content = tool.loading_data("new_simple.csv", eng=False, num=False, punc=False)  # 테스트 data read
for i in range(len(test_title)):  # teset_ title은 예측을 하는것이기 때문에 모두 지웁니다.
    test_title[i] = ""
word_to_ix, ix_to_word = tool.make_dict_all_cut(title + contents + test_content, minlength=0, maxlength=3,
                                                jamo_delete=True)  # 단어들을 인덱스화 합니다.(워드에서 인덱스 딕셔너리, 인덱스에서 워드 딕셔너리)
input_title_content = title + contents + test_content
# 단어 벡터를 분석해볼 임의의 문장들
# 문장을 전부 합친 후 공백으로 단어들을 나누고 고유한 단어들로 리스트를 만듭니다.
word_sequence = " ".join(input_title_content).split()
word_list = " ".join(input_title_content).split()
word_list = list(set(word_list))
word_list.append('<PAD>')  # make_dict_all_cut 에서 추가 되므로 Word2Vec에 들어가는 단어 리스트에도 아래의 단어들을 추가해줍니다.
word_list.append('<S>')
word_list.append('<E>')
word_list.append('<UNK>')
# 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
# 리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해,
# 이를 표현하기 위한 연관 배열과, 단어 리스트에서 단어를 참조 할 수 있는 인덱스 배열을 만듭합니다.
word_dict = {w: i for i, w in enumerate(word_list)}
skip_grams = []

for i in range(1, len(word_sequence) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    # 스킵그램을 만든 후, 저장은 단어의 고유 번호(index)로 저장합니다
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])


# skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)
    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word
    return random_inputs, random_labels


#########
# 옵션 설정
######
# 학습을 반복할 횟수
training_epoch = 10000
# 학습률
learning_rate = 0.01
# 한 번에 학습할 데이터의 크기
batch_size = 1000
# 단어 벡터를 구성할 임베딩 차원의 크기
# 이 예제에서는 x, y 그래프로 표현하기 쉽게 2 개의 값만 출력하도록 합니다.
embedding_size = 4
# word2vec 모델을 학습시키기 위한 nce_loss 함수에서 사용하기 위한 샘플링 크기
# batch_size 보다 작아야 합니다.
num_sampled = 20
# 총 단어 갯수
voc_size = len(word_list)
# print(voc_size) # 4275


#########
# 신경망 모델 구성
######
inputs = tf.placeholder(tf.int32, shape=[batch_size])
# tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [batch_size, 1] 구성해야합니다.
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
# 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
# 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

# nce_loss 함수에서 사용할 변수들을 정의합니다.
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
# 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#########
# 신경망 모델 학습
######
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    # matplot 으로 출력하여 시각적으로 확인해보기 위해
    # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
    trained_embeddings = embeddings.eval()

#########
# 임베딩된 Word2Vec 결과 확인
# 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줍니다.
######
my_dict = {}
word_to_vector = {}

for i, label in enumerate(word_list):
    # print([i, label])

    x = trained_embeddings[i]
    my_dict[label] = list(x)
    word_to_vector[label[:3]] = list(x)  # 3 글자로 바인딩 하기 위함

# print(my_dict)
# parameters
multi = True
forward_only = False
hidden_size = 300
vocab_size = len(ix_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 100
decoder_size = tool.check_doclength(title, sep=True)  # (Maximum) number of time steps in this batch
steps_per_checkpoint = 10

# transform data
encoderinputs, decoderinputs, targets_, targetweights = \
    tool.make_inputs(contents, title, word_to_ix,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)
test_encoderinputs, test_decoderinputs, test_targets_, test_targetweights = \
    tool.make_inputs(test_content, test_title, word_to_ix,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)

# print(encoderinputs)
for list_index in encoderinputs:
    for index in list_index:
        word = ix_to_word[index][:3]
#         print(word, word_to_vector[word])  # 3글자만 보게하자
# print("decode")
for list_index in decoderinputs:
    for index in list_index:
        word = ix_to_word[index][:3]
        # print(word, word_to_vector[word])  # 3글자만 보게하자

learning_rate = 0.001
n_hidden = 300
total_epoch = 500
# 입력과 출력의 형태가 one-hot 인코딩으로 같으므로 크기도 같다.
n_class = n_input = vocab_size
enc_input = tf.placeholder(tf.float32, [None, None, embedding_size])
dec_input = tf.placeholder(tf.float32, [None, None, embedding_size])
# [batch size, time steps]
targets = tf.placeholder(tf.int32, [None, None])

# 인코더 셀을 구성한다.
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더 셀을 구성한다.
with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # Seq2Seq 모델은 인코더 셀의 최종 상태값을
    # 디코더 셀의 초기 상태값으로 넣어주는 것이 핵심.
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model, labels=targets))  # sparse_softmax_cross_entropy_with_logits

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # AdamOptimizer 최적화 방법

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start = 0
end = batch_size
encoder_inputs, decoder_inputs, targets_vector, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                                 decoderinputs[start:end],
                                                                                 targets_[start:end],
                                                                                 targetweights[
                                                                                 start:end])  # 인코더 데이터, 디코터 데이터, 타겟 데이터를 가져옵니다.
encoder_vector = []
decoder_vector = []  # 위 데이터는 transpose해서 사용해주어야합니다.
for i in range(batch_size):  # 임베딩 해준거
    temp_encoder = []
    temp_decoder = []
    for j in range(encoder_size):
        temp_word = ix_to_word[encoder_inputs[j][i]][:3]

        temp_encoder.append(word_to_vector[temp_word])
    for j in range(decoder_size):
        temp_word = ix_to_word[decoder_inputs[j][i]][:3]
        temp_decoder.append(word_to_vector[temp_word])
    encoder_vector.append(np.array(temp_encoder))
    decoder_vector.append(np.array(temp_decoder))

targets_vector = np.transpose(targets_vector)
# input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):  # 학습 시작
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: encoder_vector,
                                  dec_input: decoder_vector,
                                  targets: targets_vector})
    if epoch % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


def training_result(num=0):  # 학습 결과 확인
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    start = num
    end = num + 1
    encoder_inputs, decoder_inputs, targets_vector, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                                     decoderinputs[start:end],
                                                                                     targets_[start:end],
                                                                                     targetweights[start:end])
    encoder_vector = []
    decoder_vector = []
    temp_encoder = []
    temp_decoder = []
    for j in range(encoder_size):
        temp_word = ix_to_word[encoder_inputs[j][0]][:3]

        temp_encoder.append(word_to_vector[temp_word])
    for j in range(decoder_size):
        temp_word = ix_to_word[decoder_inputs[j][0]][:3]
        temp_decoder.append(word_to_vector[temp_word])
    encoder_vector.append(np.array(temp_encoder))
    decoder_vector.append(np.array(temp_decoder))

    targets_vector = np.transpose(targets_vector)

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: encoder_vector,
                                 dec_input: decoder_vector,
                                 targets: targets_vector})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    # print(result[0])
    result_target = ""
    predict_target = ""
    training_resultd = ""
    for target_index in targets_vector[0]:
        result_target += ix_to_word[target_index]
        result_target += " "
    for result_index in result[0]:
        predict_target += ix_to_word[result_index]
        predict_target += " "
        training_resultd = (str(num) + "\ntarget : " + result_target + "\npredict : " + predict_target)

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    # end = decoded.index('E')
    return training_resultd
    # training_resultd = ''.join(decoded[:end])


def test(num=0):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    start = num
    end = num + 1
    encoder_inputs, decoder_inputs, targets_vector, target_weights = tool.make_batch(test_encoderinputs[start:end],
                                                                                     test_decoderinputs[start:end],
                                                                                     test_targets_[start:end],
                                                                                     test_targetweights[start:end])
    encoder_vector = []
    decoder_vector = []
    temp_encoder = []
    temp_decoder = []
    for j in range(encoder_size):
        temp_word = ix_to_word[encoder_inputs[j][0]][:3]

        temp_encoder.append(word_to_vector[temp_word])
    for j in range(decoder_size):
        temp_word = ix_to_word[decoder_inputs[j][0]][:3]
        temp_decoder.append(word_to_vector[temp_word])
    encoder_vector.append(np.array(temp_encoder))
    decoder_vector.append(np.array(temp_decoder))

    targets_vector = np.transpose(targets_vector)

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: encoder_vector,
                                 dec_input: decoder_vector,
                                 targets: targets_vector})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    # print(result[0])
    result_target = ""
    predict_target = ""
    training_resultd = ""
    # for target_index in targets_vector[0]:
    #     result_target += ix_to_word[target_index]
    #     result_target += " "
    for result_index in result[0]:
        predict_target += ix_to_word[result_index]
        predict_target += " "
    training_resultd = ("Test " + str(num) + "\ntarget : " + predict_target)

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    # end = decoded.index('E')
    return training_resultd


print('\n===트레이닝 결과 ===')
for i in range(batch_size):
    print(training_result(i))
print('\n===테스트 ===')
for i in range(4):
    print(test(i))

# print('word ->', training_result())
