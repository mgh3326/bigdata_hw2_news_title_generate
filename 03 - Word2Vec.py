#####################################################################
# Word2Vec 모델을 간단하게 구현해봅니다.
import tensorflow as tf
import numpy as np
import tool as tool
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
for i, label in enumerate(word_list):
    print([i, label])

    x = trained_embeddings[i]
    my_dict[label] = list(x)
print(my_dict)
#     plt.scatter(x, y)
#     plt.annotate(label, xy=(x, y), xytext=(5, 2),
#                  textcoords='offset points', ha='right', va='bottom')
#
# plt.show()
#####################################################################
# parameters
multi = True
forward_only = False
hidden_size = 300
vocab_size = len(ix_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 4
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
encoderinputs2, decoderinputs2, targets_2, targetweights2 = \
    tool.make_inputs(contents, title, my_dict,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)
test_encoderinputs2, test_decoderinputs2, test_targets_2, test_targetweights2 = \
    tool.make_inputs(test_content, test_title, my_dict,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)

class seq2seq(object):

    def __init__(self, multi, hidden_size, num_layers, forward_only,
                 learning_rate, batch_size,
                 vocab_size, encoder_size, decoder_size):

        # variables
        self.source_vocab_size = vocab_size
        self.target_vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        # networks
        W = tf.Variable(tf.random_normal([hidden_size, vocab_size]))
        b = tf.Variable(tf.random_normal([vocab_size]))
        output_projection = (W, b)
        # init_embeds = tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)
        # embeddings = tf.Variable(init_embeds)
        # self.encoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in
        #                        range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        # # self.encoder_inputs = tf.nn.embedding_lookup(embeddings, train_inputs)
        # # enc_input = tf.placeholder(tf.float32, [None, None, encoder_size])
        # # dec_input = tf.placeholder(tf.float32, [None, None, decoder_size])
        # # # [batch size, time steps]
        # # targets = tf.placeholder(tf.int64, [None, None])
        # # self.encoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in
        # #                        range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        #
        # self.decoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        # self.targets = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        # self.target_weights = [tf.placeholder(tf.float32, [batch_size]) for _ in range(decoder_size)]
        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, encoder_size])
        self.decoder_inputs = tf.placeholder(tf.float32, [None, None,decoder_size])
        self.targets = tf.placeholder(tf.int64, [None,None, decoder_size])
        self.target_weights = tf.Variable(tf.float32, [None,None, decoder_size])
        # models
        if multi:
            single_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            # cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

        if not forward_only:
            self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(  # attention
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=False)

            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]
            self.loss = []
            for logit, target, target_weight in zip(self.logits, self.targets, self.target_weights):
                crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
                self.loss.append(crossentropy * target_weight)
            self.cost = tf.add_n(self.loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        else:
            self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=True)
            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]

    def step(self, session, encoderinputs, decoderinputs, targets, targetweights, forward_only):
        input_feed = {}
        for l in range(len(encoder_inputs)):
            input_feed[self.encoder_inputs[l].name] = encoderinputs[l]
        for l in range(len(decoder_inputs)):
            input_feed[self.decoder_inputs[l].name] = decoderinputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = targetweights[l]
        if not forward_only:
            output_feed = [self.train_op, self.cost]
        else:
            output_feed = []
            for l in range(len(decoder_inputs)):
                output_feed.append(self.logits[l])

        output = session.run(output_feed, input_feed)

        if not forward_only:
            return output[1]  # loss
        else:
            return output[0:]  # outputs


sess = tf.Session()
model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers,
                learning_rate=learning_rate, batch_size=batch_size,
                vocab_size=vocab_size,
                encoder_size=encoder_size, decoder_size=decoder_size,
                forward_only=forward_only)
sess.run(tf.global_variables_initializer())  # 여기서 Saver를 이용해서 Load하는게 필요
step_time, loss = 0.0, 0.0
current_step = 0
start = 0
end = batch_size
index = 0

while current_step < 10000001:

    if end > len(title):
        start = 0
        end = batch_size
    if index > len(test_title):
        index = 0

    # Get a batch and make a step
    start_time = time.time()
    encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                              decoderinputs[start:end],
                                                                              targets_[start:end],
                                                                              targetweights[start:end])

    if current_step % steps_per_checkpoint == 0:
        for i in range(decoder_size - 2):
            decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
        output_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        predict = ' '.join(ix_to_word[ix] for ix in predict)
        real = [word[0] for word in targets]
        real = ' '.join(ix_to_word[ix] for ix in real)
        print('\n----\n step : %s \n time : %s \n LOSS : %s \n 예측 : %s \n 손질한 정답 : %s \n 정답 : %s \n----' %
              (current_step, step_time, loss, predict, real, title[start]))
        loss, step_time = 0.0, 0.0
    if (current_step) % 100 == 0:
        _encoder_inputs, _decoder_inputs, _targets, _target_weights = tool.make_batch(
            test_encoderinputs * 4,
            test_decoderinputs * 4,
            test_targets_ * 4,
            test_targetweights * 4)
        # for i in range(decoder_size - 2):
        #     _decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
        output_logits = model.step(sess, _encoder_inputs, _decoder_inputs, _targets, _target_weights, True)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        predict = ' '.join(ix_to_word[ix] for ix in predict)
        real = [word[0] for word in _encoder_inputs]
        real = ' '.join(ix_to_word[ix] for ix in real)
        print("Test predict :%s \n본문 : %s" % (predict, real))
        index += 1
    step_loss = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, False)
    step_time += time.time() - start_time / steps_per_checkpoint
    loss += np.mean(step_loss) / steps_per_checkpoint
    current_step += 1
    start += batch_size
    end += batch_size
