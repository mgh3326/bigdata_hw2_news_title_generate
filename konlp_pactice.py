from konlpy.tag import Kkma
from konlpy.utils import pprint

import numpy as np
import tensorflow as tf

import tool as tool

# data loading
data_path = './sample.csv'
title, contents = tool.loading_data(data_path, eng=False, num=False, punc=False)  # 트레이닝 data read
kkma = Kkma()
pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
a = []
for i in contents:
    a.append(kkma.nouns(i))
pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
