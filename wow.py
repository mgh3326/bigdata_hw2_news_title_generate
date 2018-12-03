#####################################################################
# Word2Vec 모델을 간단하게 구현해봅니다.
import os

import tensorflow as tf
import numpy as np
import tool as tool
from model import Seq2Seq

import time

# data loading
data_path = './sample.csv'
title, contents = tool.loading_data(data_path, eng=False, num=False, punc=False)
test_title, test_content = tool.loading_data("new_simple.csv", eng=False, num=False, punc=False)
for i in range(len(test_title)):
    test_title[i] = ""
word_to_ix, ix_to_word = tool.make_dict_all_cut(title + contents + test_content, minlength=0, maxlength=3,
                                                jamo_delete=True)
input = title + contents + test_content
# 단어 벡터를 분석해볼 임의의 문장들
# 문장을 전부 합친 후 공백으로 단어들을 나누고 고유한 단어들로 리스트를 만듭니다.
word_sequence = " ".join(input).split()
# print(word_sequence)
word_list = " ".join(input).split()
word_list = list(set(word_list))
word_list.append('<PAD>')
word_list.append('<S>')
word_list.append('<E>')
word_list.append('<UNK>')
# 문자열로 분석하는 것 보다, 숫자로 분석하는 것이 훨씬 용이하므로
# 리스트에서 문자들의 인덱스를 뽑아서 사용하기 위해,
# 이를 표현하기 위한 연관 배열과, 단어 리스트에서 단어를 참조 할 수 있는 인덱스 배열을 만듭합니다.
word_dict = {w: i for i, w in enumerate(word_list)}
# print(word_dict)
skip_grams = []

for i in range(1, len(word_sequence) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    # 스킵그램을 만든 후, 저장은 단어의 고유 번호(index)로 저장합니다
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
    # print(context)
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
        print(data[i][1])
    return random_inputs, random_labels


#########
# 옵션 설정
######
# 학습을 반복할 횟수
training_epoch = 300
# 학습률
learning_rate = 0.1
# 한 번에 학습할 데이터의 크기
batch_size = 50
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
    print([i, label])

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
print(encoderinputs)
for list_index in encoderinputs:
    for index in list_index:
        word = ix_to_word[index][:3]
        print(word, word_to_vector[word])  # 3글자만 보게하자
print("decode")
for list_index in decoderinputs:
    for index in list_index:
        word = ix_to_word[index][:3]
        print(word, word_to_vector[word])  # 3글자만 보게하자


def train(batch_size=2, epoch=100):
    model = Seq2Seq(vocab_size)

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state("./model2")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("./logs", sess.graph)
        step_time, loss = 0.0, 0.0
        current_step = 0
        start = 0
        end = batch_size
        encoder_vector = []
        decoder_vector = []

        while current_step < 10000001:

            if end > len(title):
                start = 0
                end = batch_size

            # Get a batch and make a step
            start_time = time.time()
            encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                                      decoderinputs[start:end],
                                                                                      targets_[start:end],
                                                                                      targetweights[start:end])
            for batch_size_ in range(batch_size):  # 임베딩 해준거
                temp_encoder = []
                temp_decoder = []
                for j in range(encoder_size):
                    temp_word = ix_to_word[encoder_inputs[j][batch_size_]][:3]

                    temp_encoder.append(word_to_vector[temp_word])
                for j in range(decoder_size):
                    temp_word = ix_to_word[decoder_inputs[j][batch_size_]][:3]
                    temp_decoder.append(word_to_vector[temp_word])
                encoder_vector.append(np.array(temp_encoder))
                decoder_vector.append(np.array(temp_decoder))
            targets_vector = np.transpose(targets)

            #         temp_word = ix_to_word[decoder_inputs[j][i]][:3]
            #         temp_decoder.append(word_to_vector[temp_word])
            #     encoder_vector.append(np.array(temp_encoder))
            #     decoder_vector.append(np.array(temp_decoder))
            # for i in range(encoder_size):
            #
            #     temp_encoder = []
            #     for j in range(batch_size):
            #         temp_word = ix_to_word[encoder_inputs[i][j]][:3]
            #         temp_encoder.append(word_to_vector[temp_word])
            #     encoder_vector.append(np.array(temp_encoder))
            #
            # for i in range(decoder_size):
            #     temp_decoder = []
            #     for j in range(batch_size):
            #         temp_word = ix_to_word[decoder_inputs[i][j]][:3]
            #         temp_decoder.append(word_to_vector[temp_word])
            #     decoder_vector.append(np.array(temp_decoder))

            # for step in range(total_batch * epoch):
            #     enc_input, dec_input, targets = dialog.next_batch(batch_size)

            _, loss = model.train(sess, encoder_vector, decoder_vector, targets_vector)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, encoder_inputs, decoder_inputs, targets_vector)

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join("./model2", "news.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')


# def test(dialog, batch_size=100):
#     print("\n=== 예측 테스트 ===")
#
#     model = Seq2Seq(dialog.vocab_size)
#
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
#         print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
#         model.saver.restore(sess, ckpt.model_checkpoint_path)
#
#         enc_input, dec_input, targets = dialog.next_batch(batch_size)
#
#         expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)
#
#         expect = dialog.decode(expect)
#         outputs = dialog.decode(outputs)
#
#         pick = random.randrange(0, len(expect) / 2)
#         input = dialog.decode([dialog.examples[pick * 2]], True)
#         expect = dialog.decode([dialog.examples[pick * 2 + 1]], True)
#         outputs = dialog.cut_eos(outputs[pick])
#
#         print("\n정확도:", accuracy)
#         print("랜덤 결과\n")
#         print("    입력값:", input)
#         print("    실제값:", expect)
#         print("    예측값:", ' '.join(outputs))
#


train()
