import tensorflow as tf
import numpy as np
import tool as tool
import time

# data loading
data_path = './test.csv'
title, contents = tool.loading_data(
    data_path, eng=False, num=False, punc=False)
word_to_ix, ix_to_word = tool.make_dict_all_cut(
    title + contents, minlength=0, maxlength=3, jamo_delete=True)
print("test")
