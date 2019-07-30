from kenn.AttCNN import AttCNN
from kenn.utils import load_data


def main():
    train_x, train_y, test_x, test_y = load_data('weak_two')
    model = AttCNN(train_x, train_y, test_x, test_y,
                   seg_len=650, num_channels=3, num_labels=9,
                   num_conv_for_extract=2, filters=16, k_size=5, conv_strides=1, pool_size=4, pool_strides=4,
                   batch_size=100, learning_rate=0.0001, num_epochs=1000,
                   print_val_each_epoch=2, print_test_each_epoch=10, print_test=True,
                   cpt_func='dot', norm_func='softmax', padding='same',
                   att_cnn_filters1=64, att_cnn_filters2=64, att_cnn_filters3=64,
                   cnn_type='1d', bool_bn=False, bool_visual_att=True, act_func='relu',
                   no_exp=1)
    model.train()


if __name__ == '__main__':
    main()
