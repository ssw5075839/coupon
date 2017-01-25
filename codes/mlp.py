import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset import Dataset
from sklearn.metrics import log_loss
import mxnet as mx
import os
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Implement data processing pipeline and MLP classifier for Coupon Puchase Prediction')
    parser.add_argument('--data_root', type=str, default='/home/shiwei/kaggle-coupon-purchase-prediction/data',
                        help='the path to where data is located')
    parser.add_argument('--NEGA_WEIGHT', type=int, default=2,
                        help='the ratio that how many negative coupons will be sampled')
    parser.add_argument('--N_EPOCH', type=int, default=150,
                        help='the number of epoch within which the MLP will be trained')
    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='the batch size the MLP will use')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default is 0.01')
    parser.add_argument('--lr_factor_epoch', type=int, default=100,
                        help='learning rate decrease period in epoch, default is 100')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='learning rate decrease factor, default is multiplied by 0.5')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of SGD optimizer, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='L2 regularization coefficient add to all the weights, default is 0.0001')
    return parser.parse_args()
    
def logloss(label,pred):
    return log_loss(label.flatten(),pred.flatten(),eps=1e-6)
    
preds = []
scaler = StandardScaler()
def callback(rec):
    feats = rec["coupon_feats"]
    pred = np.zeros(len(feats), dtype=np.float32)
    
    for i in range(len(models)):
        pred += models[i].predict(scaler.transform(feats)).reshape(pred.shape)
    pred /= len(models)
    
    scores = zip(pred, rec["coupon_ids"])
    scores = sorted(scores, key = lambda score: -score[0])
    coupon_ids = " ".join(map(lambda score: str(score[1]), scores[0:10]))
    preds.append([rec["user_id"], coupon_ids])

if __name__ == '__main__':
    args = parse_args()
    data_root = args.data_root
    NEGA_WEIGHT = args.NEGA_WEIGHT
    N_EPOCH = args.N_EPOCH
    BATCH_SIZE = args.BATCH_SIZE
    lr = args.lr
    lr_factor_epoch = args.lr_factor_epoch
    lr_factor = args.lr_factor
    momentum = args.momentum
    wd = args.wd
    
    print"start to process raw data and feature engineering:"
    if os.path.isfile(data_root+"/all_data.pkl"):
        dataset = Dataset.load_pkl(data_root+"/all_data.pkl")
    else:
        dataset = Dataset(datadir=data_root)
        dataset.load()
        dataset.save_pkl(data_root+"/all_data.pkl")
        dataset = Dataset.load_pkl(data_root+"/all_data.pkl")

    # estimate mean,std
    print"start to gather train data:"
    np.random.seed(seed=71)
    x, y_train = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
    scaler.fit(x)

    models = []
    # logging
    head = '%(asctime)-15s Node[' + str(0) + '] %(message)s'
    log_file = 'mlp_mxnet.log'
    data_dir = data_root
    log_dir = data_dir
    log_file_full_name = os.path.join(log_dir, log_file)
    if not os.path.exists(log_dir): 
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_file_full_name,filemode='w')
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(head) 
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    seeds = [74,142,318,1120,211,350,455]
    # You can also fill in your lucky number here
    for i,s in enumerate(seeds):
        # resampling the training dataset
        np.random.seed(seed=s)
        x, y_train = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
        x_train = scaler.transform(x)
        #print x_train.shape
        # logging
        kv = mx.kvstore.create('local')    
        train_iterator = mx.io.NDArrayIter(x_train, y_train.flatten(), BATCH_SIZE,
                                     shuffle = True, last_batch_handle='roll_over')

        print "data preparation done!"

        data = mx.symbol.Variable('data')
        fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=512)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
        do1  = mx.symbol.Dropout(data = act1, name='do1', p=0.5)
        fc2  = mx.symbol.FullyConnected(data = do1, name = 'fc2', num_hidden = 32)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        do2  = mx.symbol.Dropout(data = act2, name='do2', p=0.1)
        fc3  = mx.symbol.FullyConnected(data = do2, name='fc3', num_hidden=1)
        mlp  = mx.symbol.LogisticRegressionOutput(data = fc3, name='softmax')

        model_args = {}
        epoch_size = x_train.shape[0] / BATCH_SIZE
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                    step = max(int(epoch_size * lr_factor_epoch), 1),
                    factor = lr_factor)
        model = mx.model.FeedForward(ctx = mx.gpu(), symbol=mlp, num_epoch=N_EPOCH, 
                learning_rate=lr,momentum=momentum, wd=wd,
                initializer=mx.initializer.Xavier(rnd_type='gaussian',factor_type="avg",magnitude=2),
                **model_args)

        print "start to train:"
        logger.info('start with arguments %s', 'ssw')
        logloss_eval_metric = mx.metric.CustomMetric(logloss, name='logloss')

        model.fit(X=train_iterator,kvstore=kv,eval_metric=logloss_eval_metric,
                  batch_end_callback=mx.callback.log_train_metric(int(x_train.shape[0]/BATCH_SIZE/2)),logger=logger)
        model.save(data_root+'/mlp_seeds_'+str(s))
        models.append(model)
        del kv
    #start to predict
    print "start to predict:"
    dataset.each_test(callback)
    preds = sorted(preds, key=lambda rec: rec[0])
    fp = open("submission_mlp_mxnet_preview_feature.csv", "w")
    fp.write("USER_ID_hash,PURCHASED_COUPONS\n")
    for pred in preds:
        fp.write("%s,%s\n" % (pred[0], pred[1]))
    fp.close()