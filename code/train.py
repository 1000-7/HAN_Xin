# coding=utf-8

"""
xin
"""

import tensorflow as tf
import model
import time
import os
from load_data import read_dataset, batch_iter
from tensorflow.python.framework import graph_util
from config import config

config = config["10"]
def train_model(train_data_path,calss_num,model_path):
    # 晴空计算图
    tf.reset_default_graph()
    # Data loading params
    tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
    tf.flags.DEFINE_integer("vocab_size",config["vocab_size"], "vocabulary size")
    tf.flags.DEFINE_integer("num_classes", calss_num, "number of classes")
    tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
    tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
    tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")  # 防止梯度爆炸

    FLAGS = tf.flags.FLAGS
    # Use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    configenv = tf.ConfigProto()

    configenv.gpu_options.per_process_gpu_memory_fraction = 0.6
    configenv.gpu_options.allow_growth = True

    with tf.Session(config = configenv) as sess:
        han = model.HAN(vocab_size=FLAGS.vocab_size,
                        num_classes=FLAGS.num_classes,
                        embedding_size=FLAGS.embedding_size,
                        hidden_size=FLAGS.hidden_size)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,
                                                                          logits=han.out,
                                                                          name='loss'))
        with tf.name_scope('accuracy'):
            predict = tf.argmax(han.out, axis=1, name='predict')
            label = tf.argmax(han.input_y, axis=1, name='label')
            acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar('loss', loss)
        acc_summary = tf.summary.scalar('accuracy', acc)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.max_sentence_num: 30,
                han.max_sentence_length: 30,
                han.batch_size: 64
            }
            _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

            time_str = str(int(time.time()))
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return step


        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
                han.input_x: x_batch,
                han.input_y: y_batch,
                han.max_sentence_num: 30,
                han.max_sentence_length: 30,
                han.batch_size: 64
            }
            step_now, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
            time_str = str(int(time.time()))
            print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step_now, cost,
                                                                                               accuracy))
            if writer:
                writer.add_summary(summaries, step_now)


        for epoch in range(FLAGS.num_epochs):
            print('current epoch %s' % (epoch + 1))
            train_x, train_y, dev_x, dev_y = read_dataset(train_data_path)
            print("data load finished")
            for i in range(0, 200000, FLAGS.batch_size):
                x = train_x[i:i + FLAGS.batch_size]
                y = train_y[i:i + FLAGS.batch_size]
                try:
                    step = train_step(x, y)
                except ValueError:
                    continue
                if step % FLAGS.evaluate_every == 0:
                    try:
                        dev_step(dev_x, dev_y, dev_summary_writer)
                    except ValueError:
                        continue

        # 写入序列化的pb文件
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            output_node_names = ["doc_classification/predict"] #< -- 参数：output_node_names，输出节点名
        )
        with tf.gfile.GFile(model_path + '.pb', mode='wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)
    print("训练完成")
if __name__ == "__main__":
    # train_model("../traindata/main_model/",11,"../model/model-11")
    train_model("../traindata/3_6_7_submodel/", 3, "../model/model-3-6-7")
    # train_model("../traindata/1_2_submodel/", 2, "../model/model-1-2")