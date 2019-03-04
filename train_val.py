from network import *

# 训练和验证数据
train_file = 'train.tfrecords'
val_file = 'val.tfrecords'
# 图片数量
train_num = 600*69
val_num = 200*69
# 训练多少轮
EPOCH = 50
# 每次batch大小
BATCH = 200
# 可变学习率
Learn_rate = [0.001, 0.0005, 0.0001]
# 学习率变化发生在第几轮
change_step = [20, 50]
# 模型保存位置
model_save_dir = "save"
# 日志文件存储位置
summaries_save_dir = 'tmp/data'


def main():
    # 装载数据模块 调用前面的文件传输函数 同时再次将图片顺序打乱
    with tf.name_scope("load_data"):
        train_img, train_lab = data_loader(train_file)
        train_image, train_label = tf.train.shuffle_batch([train_img, train_lab], batch_size=BATCH,
                                                          capacity=1005, min_after_dequeue=1000)
        val_img, val_lab = data_loader(val_file)
        val_image, val_label = tf.train.shuffle_batch([val_img, val_lab], batch_size=BATCH, capacity=1005,
                                                      min_after_dequeue=1000)
    # 输入, 包含图片,标签,dropout, 学习率,都使用占位符代替
    with tf.name_scope('input'):
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3])
        correct_label = tf.placeholder(dtype=tf.float32, shape=[None, students_num])
        learn_rate = tf.placeholder(dtype=tf.float32)
        drop_input = tf.placeholder(dtype=tf.float32)
    # 网络预测,调用network
    predict, regular = network(input_image, drop_input)
    # 计算准确率
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(correct_label, 1), tf.argmax(predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.summary.scalar('accuracy', accuracy)
    # 计算交叉熵损失函数
    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(correct_label * tf.log(tf.clip_by_value(predict, 1e-10, 1.0))) + regular
        tf.summary.scalar('loss', loss)
    # 训练步骤,使用Adam自适应学习率的优化器
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        tf.summary.scalar('learn_rate', learn_rate)
    # 将日志文件打包合并
    merged = tf.summary.merge_all()
    # 开启一个会话
    sess = tf.InteractiveSession()
    # 添加日志
    summary_writer = tf.summary.FileWriter(summaries_save_dir, sess.graph)
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # 开启线程装载数据
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)
    # 记录最好的准确率及其对应的轮次和loss
    count = 0
    max_accuracy = 0
    relate_loss = 0
    relate_epoch = 0
    # 训练
    for i in range(EPOCH):
        print(i, "th is train...")
        # 确定学习率
        if 0 <= i <= change_step[0]:
            rate = Learn_rate[0]
        else:
            if change_step[0] < i <= change_step[1]:
                rate = Learn_rate[1]
            else:
                rate = learn_rate[2]
        for j in range(int(train_num / BATCH)):
            count += 1
            # 获取图片及标签
            image, label = sess.run([train_image, train_label])
            # 没运行100个batch就记录一次日志, 否则只进行训练
            if count % 100 == 0:
                summary, train_1 = sess.run([merged, train_step],
                                            feed_dict={input_image: image, correct_label: label,
                                                       learn_rate: rate, drop_input: 0.5})
                summary_writer.add_summary(summary, count)
            else:
                # 只进行训练,训练时dropout使用0.5
                train_step.run(feed_dict={input_image: image, correct_label: label,
                                          learn_rate: rate, drop_input: 0.5})
        # 每训练一轮就相应的运行一次验证集, 以此为依据调整超参数
        mean_loss = 0
        mean_acc = 0
        mean_regu = 0
        for j in range(int(val_num / BATCH)):
            image, label = sess.run([val_image, val_label])
            accuracy_i, regu, loss_i, = sess.run([accuracy, regular, loss],
                                                 feed_dict={input_image: image, correct_label: label,
                                                            drop_input: 1.0})
            mean_acc += accuracy_i
            mean_loss += loss_i
            mean_regu += regu
        mean_loss = mean_loss * BATCH / val_num
        mean_acc = mean_acc * BATCH / val_num
        mean_regu = mean_regu * BATCH / val_num
        # 如果当前为最好的准确率,保存当前模型
        if max_accuracy < mean_acc:
            max_accuracy = mean_acc
            relate_loss = mean_loss
            relate_epoch = i
            save_net(sess, model_save_dir+'/best.ckpt')
        # 输出当前结果以及历史最好的结果
        print('Epoch: ', i, "  ", "Accuracy: ", mean_acc, "  ", "Loss: ", mean_loss, "regulater: ", mean_regu)
        print('Best Epoch: ', relate_epoch, "  ", 'Accuracy: ', max_accuracy, '  Loss: ', relate_loss)
        print('\n')
        # 每10次保存一个模型
        if i % 10 == 0:
            save_net(sess, model_save_dir + '/' + str(i) + '.ckpt')
    save_net(sess, model_save_dir+'/lastmodel.ckpt')
    # 关闭日志文件
    summary_writer.close()


if __name__ == '__main__':
    main()
