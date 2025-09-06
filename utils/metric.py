import numpy as np
from multiprocessing import Pool
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class ConfusionMatrix(object):
    """Simple confusion matrix accumulator."""
    def __init__(self, nclass, classes=None, ignore_label=255):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass), dtype=np.float64)
        self.ignore_label = ignore_label

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == self.ignore_label:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            denom = np.sum(self.M[:, i])
            if denom != 0:
                recall += self.M[i, i] / denom
        return recall / self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            denom = np.sum(self.M[i, :])
            if denom != 0:
                accuracy += self.M[i, i] / denom
        return accuracy / self.nclass

    def jaccard(self):
        jaccard_perclass = []
        for i in range(self.nclass):
            denom = (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i])
            if denom != 0:
                jaccard_perclass.append(self.M[i, i] / denom)
        if len(jaccard_perclass) == 0:
            return 0.0, [], self.M
        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M


def _generateM_static(item):
    """Top-level helper for multiprocessing: item = (gt_flat, pred_flat, nclass)"""
    gt, pred, nclass = item
    m = np.zeros((nclass, nclass), dtype=np.float64)
    L = len(gt)
    for i in range(L):
        g = int(gt[i])
        p = int(pred[i])
        if 0 <= g < nclass and 0 <= p < nclass:
            m[g, p] += 1.0
    return m


def get_iou(data_list, class_num, save_path=None):
    """Compute mean IoU using multiprocessing safely.

    data_list: iterable of (gt_flat, pred_flat)
    class_num: number of classes
    """
    ConfM = ConfusionMatrix(class_num)
    items = [(gt, pred, class_num) for (gt, pred) in data_list]
    pool = Pool()
    try:
        m_list = pool.map(_generateM_static, items)
    finally:
        pool.close()
        pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
    return aveJ, j_list


def metrics_from_confusion(M):
    """Compute precision, recall, f1, accuracy from a confusion matrix M (C x C).

    Returns (precision_mean, recall_mean, f1_mean, accuracy_overall, precision_per, recall_per, f1_per)
    """
    M = np.array(M, dtype=np.float64)
    tp = np.diag(M)
    sum_row = M.sum(axis=1)  # actual (gt)
    sum_col = M.sum(axis=0)  # predicted

    # per-class precision, recall
    precision_per = np.divide(tp, sum_col, out=np.zeros_like(tp), where=sum_col != 0)
    recall_per = np.divide(tp, sum_row, out=np.zeros_like(tp), where=sum_row != 0)

    # per-class F1
    denom = (precision_per + recall_per)
    f1_per = np.divide(2 * precision_per * recall_per, denom, out=np.zeros_like(denom), where=denom != 0)

    # overall accuracy
    total = M.sum()
    accuracy_overall = float(tp.sum() / total) if total > 0 else 0.0

    precision_mean = float(np.mean(precision_per))
    recall_mean = float(np.mean(recall_per))
    f1_mean = float(np.mean(f1_per))

    return precision_mean, recall_mean, f1_mean, accuracy_overall, precision_per.tolist(), recall_per.tolist(), f1_per.tolist()


def calculate_metrics(y_true, y_pred=None, average="macro"):
    """
    Backwards-compatible wrapper. Accepts either flat label arrays (y_true, y_pred)
    or a confusion matrix (2D numpy array) passed as y_true (y_pred ignored).
    Returns: precision, recall, F1, accuracy
    """
    # If caller passed a confusion matrix as first arg
    y = np.array(y_true)
    if y.ndim == 2 and y.shape[0] == y.shape[1]:
        return metrics_from_confusion(y)[:4]
    # fallback to flat arrays
    return calculate_metrics_flat(y_true, y_pred, average=average)


def calculate_metrics_flat(y_true, y_pred, average="macro"):
    """Compute precision/recall/F1/accuracy from flat label lists using sklearn."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    F1_score = f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, F1_score, accuracy
