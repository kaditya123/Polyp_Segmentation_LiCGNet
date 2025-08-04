import os, sys
import numpy as np
from multiprocessing import Pool 
import copyreg
import types
import cv2
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # install the scikit-learn
from sklearn.metrics import fbeta_score


def calculate_metrics(y_true, y_pred, average="macro"):
    """
    Calculate precision, recall, F2 score, and accuracy based on true and predicted labels.
    Arguments:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - average: Average type for metrics calculation ('macro', 'micro', 'weighted')

    Returns:
    - precision, recall, F1 score, accuracy
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    F1_score = f1_score(y_true, y_pred, average=average)  # F1 score
    # F2_score = fbeta_score(y_true, y_pred, beta=2, average=average) 
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, F1_score, accuracy


def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

class ConfusionMatrix(object):
    def __init__(self, nclass, classes=None, ignore_label= 255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))
        self.ignore_label= ignore_label

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m

def get_iou(data_list, class_num, save_path=None):
    """ 
    Args:
      data_list: a list, its elements [gt, output]
      class_num: the number of label
    """
    from multiprocessing import Pool 

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')
    return aveJ, j_list

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation metrics')
    parser.add_argument('--test_ids', type=str, required=True, help='Path to file with test image ids')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted masks')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth masks')
    parser.add_argument('--class_num', type=int, required=True, help='Number of classes')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    m_list = []
    data_list = []
    test_ids = [i.strip() for i in open(args.test_ids) if not i.strip() == '']
    for index, img_id in enumerate(test_ids):
        if index % 100 == 0:
            print('%d processd'%(index))
        pred_img_path = os.path.join(args.pred_dir, img_id+'.png')
        gt_img_path = os.path.join(args.gt_dir, img_id+'.png')
        pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
        # show_all(gt, pred)
        data_list.append([gt.flatten(), pred.flatten()])

    ConfM = ConfusionMatrix(args.class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    with open(args.save_path, 'w') as f:
        f.write('meanIOU: ' + str(aveJ) + '\n')
        f.write(str(j_list)+'\n')
        f.write(str(M)+'\n')
