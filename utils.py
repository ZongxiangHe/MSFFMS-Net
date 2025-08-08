import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation1, label1, segmentation2, label2, segmentation3, label3, save_dir, decision=None, blur=True, plot_seg=False):
    #print(segmentation1.shape,label1.shape,segmentation2.shape)
    plt.figure()
    plt.clf()
    plt.subplot(1, 8, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image', fontsize=8)
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation1 = np.transpose(segmentation1)
        label1 = np.transpose(label1)
        segmentation2 = np.transpose(segmentation2)
        label2 = np.transpose(label2)
        segmentation3 = np.transpose(segmentation3)
        label3 = np.transpose(label3)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, 8, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth', fontsize=8)
    plt.imshow(label1, cmap="gray")

    plt.subplot(1, 8, 3)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('Output', fontsize=8)
    else:
        plt.title(f"Output: {decision:.5f}", fontsize=8)
    # display max
    vmax_value1 = max(1, np.max(segmentation1))
    plt.imshow(segmentation1, cmap="jet", vmax=vmax_value1)

    plt.subplot(1, 8, 4)
    plt.xticks([])
    plt.yticks([])
    # if decision is None:
    #     plt.title('Output', fontsize=8)
    # else:
    #     plt.title(f"Output: {decision:.5f}", fontsize=8)
    # display max
    vmax_value2 = max(1, np.max(segmentation2))
    plt.imshow(segmentation2, cmap="jet", vmax=vmax_value2)
    plt.subplot(1, 8, 5)
    plt.xticks([])
    plt.yticks([])
    # if decision is None:
    #     plt.title('Output', fontsize=8)
    # else:
    #     plt.title(f"Output: {decision:.5f}", fontsize=8)
    # display max
    vmax_value3 = max(1, np.max(segmentation3))
    plt.imshow(segmentation3, cmap="jet", vmax=vmax_value3)

    plt.subplot(1, 8, 6)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output scaled', fontsize=8)
    if blur:
        normed1 = segmentation1 / segmentation1.max()
        blured1 = cv2.blur(normed1, (32, 32))
        plt.imshow((blured1 / blured1.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation1 / segmentation1.max() * 255).astype(np.uint8), cmap="jet")
    plt.subplot(1, 8, 7)
    plt.xticks([])
    plt.yticks([])
    #plt.title('Output scaled')
    if blur:
        normed2 = segmentation2 / segmentation2.max()
        blured2 = cv2.blur(normed2, (32, 32))
        plt.imshow((blured2 / blured2.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation2 / segmentation2.max() * 255).astype(np.uint8), cmap="jet")

    plt.subplot(1, 8, 8)
    plt.xticks([])
    plt.yticks([])
    #plt.title('Output scaled', fontsize=8)
    if blur:
        normed3 = segmentation3 / segmentation3.max()
        blured3 = cv2.blur(normed3, (32, 32))
        plt.imshow((blured3 / blured3.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation3 / segmentation3.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''

    plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)


def evaluate_metrics(samples, results_path, run_name):
    samples = np.array(samples)

    img_names = samples[:, 4]
    predictions = samples[:, 0]
    labels = samples[:, 3].astype(np.float32)

    metrics = get_metrics(labels, predictions)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')


def get_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics
