import matplotlib.pyplot as plt
import os.path as osp
import os


def plot_features(features, labels, num_classes, epoch, prefix, legends=None):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    if legends is None:
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    else:
        plt.legend(legends, loc='upper right')
    dirname = osp.join("E:/pic", prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_{:03}.png'.format(epoch+1))
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
