
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def softmax(x,axis=0):
  return np.exp(x) / np.sum(np.exp(x), axis=axis)
def show_attention(input_src, output_words, attentions):
    # Set up figure with colorbar

    axes = plt.subplot(211)

    print input_src.shape
    axes.matshow(np.squeeze(input_src[0]), cmap='rainbow')
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_aspect('auto')
    # plt.ylabel('Entities')



    axes = plt.subplot(212)
    print input_src.shape
    cax=axes.matshow(np.squeeze(input_src[1]), cmap='rainbow')
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_aspect('auto')


    # plt.colorbar(cax)
    # ax = plt.subplot(212)
    # cax = ax.matshow(attentions, cmap='Reds')
    #plt.colorbar(cax)
    #
    # # Set up axes
    # ax.set_xticklabels([''] * attentions.shape[1], rotation=90)
    # ax.set_yticklabels([''] + output_words)
    #
    # ax.set_aspect('auto')
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # plt.ylabel('Predictions')
    # plt.xlabel('Frames')

    plt.show()
    plt.close()
def read_npybin(file_path='conv1.npy'):
    conv1= np.load(file_path)
    conv1.shape=5,512,1,512
    candidate1=conv1[:,0:10,:,0:4]

    # for i in range(10):
    #   candidate1[i]=softmax(candidate1[i],axis=1)
    #
    # candidate2 = conv1[2, 0:10, :, 0:10]
    #
    # for i in range(10):
    #   candidate2[i] = softmax(candidate2[i],axis=1)

    candidate=np.transpose(candidate1)
    # candidate=np.sum(candidate1,axis=0)
    # candidate=np.concatenate((candidate1,candidate2),axis=1)
    print candidate.shape

    show_attention(candidate,'1',None)
def recover_models(meta_path = '../models/conv_pool120_10_4_8982/model.ckpt-20700.meta',
                   model_path = '../models/conv_pool120_10_4_8982/model.ckpt-20700.data-00000-of-00001'):
  latest_checkpoint0 = tf.train.latest_checkpoint("../models/conv_pool120_10_4_8982/")

  saver0 = tf.train.import_meta_graph(meta_path)
  all_vars = tf.global_variables()
  # for v in all_vars:
  #   print v.name
  #tower/conv1/convW
# tower/conv1/convW
  v_vars = [v for v in all_vars if v.name == 'tower/conv1/convW:0'
            or v.name == 'tower/conv2/convW:0']
  if latest_checkpoint0:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      # saver=tf.train.import_meta_graph(meta_path)
      saver = tf.train.Saver(v_vars)
      saver.restore(sess, latest_checkpoint0)
      conv1=np.zeros((5,512,1,512))
      conv2=np.zeros((3,1024,1,1024))
      conv1=v_vars[0].eval()
      conv2=v_vars[1].eval()
      np.save("conv1.npy",conv1)
      np.save("conv2.npy",conv2)
  # saver = tf.train.import_meta_graph(meta_path)  # 导入图

  # config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  # with tf.Session() as sess:
  #   saver.restore(sess, model_path)  # 导入变量值
    # graph = tf.get_default_graph()
    # prob_op = graph.get_operation_by_name('prob')  # 这个只是获取了operation， 至于有什么用还不知道

    # for i in v_vars:
    #   logging.info(str(i))
  #
  # prediction = graph.get_tensor_by_name('prob:0')  # 获取之前prob那个操作的输出，即prediction
  # print(ress.run(prediciton, feed_dict={...}))  # 要想获取这个值，需要输入之前的placeholder （这里我编辑文章的时候是在with里面的，不知道为什么查看的时候就在外面了...）
  # print(
  # sess.run(graph.get_tensor_by_name('logits_classifier/weights:0')))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重
if __name__ == '__main__':
    read_npybin()
# def get_conv_layer_names():
# # Load the Inception model.
# model = inception.Inception()
#
# # Create a list of names for the operations in the graph
# # for the Inception model where the operator-type is 'Conv2D'.
# names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']
#
# # Close the TensorFlow session inside the model-object.
# model.close()
#
# return names