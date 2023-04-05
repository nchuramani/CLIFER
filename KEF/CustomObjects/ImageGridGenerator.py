import os
from matplotlib import pyplot as plt, gridspec
import numpy
import math

from scipy.misc import imread, imresize, imsave


def write_image_grid(filepath, imgs, labels=None, label_dict=None, figsize=None, cmap='gray', isGrayscale=False,
                     comment=None, dset=None, length=None):
    directory = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(directory):
        os.makedirs(directory)

    if isGrayscale:
        imgs = numpy.squeeze(imgs, axis=-1)
    if length is not None:
        save_batch_images_alt(batch_images=imgs,save_path=filepath, isGrayscale=isGrayscale, size_frame=length)
    else:
        save_batch_images_alt(batch_images=imgs, save_path=filepath, isGrayscale=isGrayscale)
    if dset is not None and dset.startswith("MEFED"):
        label_txt = numpy.array([label_dict[i[0]] for i in labels])
        numpy.savetxt(filepath + "_labels.csv",label_txt, delimiter=',',fmt='%s')



def create_image_grid(imgs, figsize=None, cmap=None, comment=None, labels=None, dict=None):
    number = imgs.shape[0]
    h = w = int(math.sqrt(number))
    if figsize is None:
        figsize = (h, w)
    fig = plt.figure(figsize=figsize)
    gs1 = gridspec.GridSpec(h, w)
    gs1.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    for i in range(0,h):
        for j in range(0,w):
            ax = plt.subplot(gs1[i, j])
            pos = i*h + j
            img = imgs[pos, :]
            if dict is not None:
                ax.set_title(str(dict[int(labels[pos])]), y=-0.01,fontsize=10)
            ax.imshow((img * 127.5 + 127.5).astype(numpy.uint8), cmap=cmap)
            ax.axis('off')
    if comment is not None:
        fig.suptitle('Discriminator Img Acc: ' + str(comment))
    return fig


def save_batch_images_alt(batch_images, save_path, image_value_range=(-1, 1), size_frame=None, isGrayscale=False):
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(numpy.ceil(numpy.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    if size_frame[0] == 1:
        size_frame[1] = len(batch_images)
    if isGrayscale:
        frame = numpy.zeros([img_h * size_frame[0], img_w * size_frame[1]])
    else:
        frame = numpy.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        if isGrayscale:
            frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w)] = image
        else:
            frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
        frame_show = frame.copy()
        frame_show[:, :, 0] = frame[:, :, 2]
        frame_show[:, :, 2] = frame[:, :, 0]
        imsave(save_path, frame_show)




