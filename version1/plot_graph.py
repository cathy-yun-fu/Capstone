import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def generate_graph(tr_loss, tr_acc, val_loss, val_acc, title, output_dir):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    x_loss = np.arange(1, len(tr_loss) + 1)
    avg_loss_diff = (val_loss - tr_loss).mean()
    ax1.plot(x_loss, tr_loss, 'r--', linewidth=2.0, label='training loss')
    ax1.plot(x_loss, val_loss, 'r', linewidth=2.0, label='val loss')
    ax1.set_xticks(x_loss)
    ax1.set_title('Loss')
    ax1.legend(shadow=True, fancybox=True)

    x_acc = np.arange(1, len(tr_acc) + 1)
    ax2.plot(x_acc, tr_acc, 'b--', linewidth=2.0, label='training acc.')
    ax2.plot(x_acc, val_acc, 'b', linewidth=2.0, label='val acc.')
    ax2.set_title('Accuracy')
    ax2.legend(shadow=True, fancybox=True)

    f.suptitle(title + ' (avg_loss_diff={:.4f})'.format(
               avg_loss_diff), fontsize=16)

    plt.savefig(os.path.join(output_dir, title + '.png'))
#
# if __name__ == '__main__':
#     generate_graph(np.array([5,4,3,2,1]), np.array([1,2,3,4,5]), np.array([4,3,2,1,0]), np.array([2,3,4,5,6]) ,'testplot','./')