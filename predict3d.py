import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
DATABASE = 'MRABrain/'
#
args = {
    'test_path': './dataset/' + DATABASE + 'test/',
    'pred_path': '/dassets/' + 'predict/'
}

if not os.path.exists(args['pred_path']):
    os.makedirs(args['pred_path'])


def load_3d():
    test_images = []
    test_labels = []
    for file in glob.glob(os.path.join(args['test_path'], 'images', '*.mha')):
        basename = os.path.basename(file)
        file_name = basename[:-8]
        image_name = os.path.join(args['test_path'], 'images', basename)
        label_name = os.path.join(args['test_path'], 'label', file_name + 'gt.mha')
        test_images.append(image_name)
        test_labels.append(label_name)
    return test_images, test_labels


def load_net():
    net = torch.load('/home/imed/Research/Attention/checkpoint/model.pkl')
    return net


def save_prediction(pred, filename='', spacing=None):
    pred = torch.argmax(pred, dim=1)
    save_path = args['pred_path'] + 'pred/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Make dirs success!")
    # for MSELoss()
    mask = (pred.data.cpu().numpy() * 255).astype(np.uint8)

    mask = mask.squeeze(0)  # for CE Loss

    mask = sitk.GetImageFromArray(mask)

    sitk.WriteImage(mask, os.path.join(save_path + filename + ".mha"))



def save_label(label, index, spacing=None):
    label_path = args['pred_path'] + 'label/'
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    label = sitk.GetImageFromArray(label)
    if spacing is not None:
        label.SetSpacing(spacing)
    sitk.WriteImage(label, os.path.join(label_path, index + ".mha"))


def predict():
    net = load_net()
    images, labels = load_3d()
    with torch.no_grad():
        net.eval()
        for i in tqdm(range(len(images))):
            name_list = images[i].split('/')
            index = name_list[-1][:-4]
            image = sitk.ReadImage(images[i])
            image = sitk.GetArrayFromImage(image).astype(np.float32)
            image = image / 255
            label = sitk.ReadImage(labels[i])
            label = sitk.GetArrayFromImage(label).astype(np.int64)
            # label = label / 255
            # VascuSynth
            # image = image[2:98, 2:98, 2:98]
            # label = label[2:98, 2:98, 2:98]
            save_label(label, index)
            # if cuda
            image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0)
            image = image.cuda()
            output = net(image)
            save_prediction(output, filename=index + '_pred', spacing=None)


if __name__ == '__main__':
    predict()
