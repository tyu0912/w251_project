import sys
sys.path.insert(1, '/temporal-shift-module/online_demo')

from mobilenet_v2_tsm_test import MobileNetV2
from PIL import Image
import urllib.request
import os
import torch
import torchvision
import numpy as np
import cv2
import time


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]



class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)



class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()



class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    #if idx_ in [7, 8, 21, 22, 1, 3]:
    #    idx_ = history[-1]

    if idx_ in [1,4,5,6]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 9
    
    # history smoothing

    #print(history[-1])
    #print(idx_)

    if idx_ != history[-1] and len(history) != 1:
        if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
            idx_ = history[-1]
    

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


def get_categories(num_classes):

    if num_classes == 27:
        catigories = [
        "Doing other things",  # 0
        "Drumming Fingers",  # 1
        "No gesture",  # 2
        "Pulling Hand In",  # 3
        "Pulling Two Fingers In",  # 4
        "Pushing Hand Away",  # 5
        "Pushing Two Fingers Away",  # 6
        "Rolling Hand Backward",  # 7
        "Rolling Hand Forward",  # 8
        "Shaking Hand",  # 9
        "Sliding Two Fingers Down",  # 10
        "Sliding Two Fingers Left",  # 11
        "Sliding Two Fingers Right",  # 12
        "Sliding Two Fingers Up",  # 13
        "Stop Sign",  # 14
        "Swiping Down",  # 15
        "Swiping Left",  # 16
        "Swiping Right",  # 17
        "Swiping Up",  # 18
        "Thumb Down",  # 19
        "Thumb Up",  # 20
        "Turning Hand Clockwise",  # 21
        "Turning Hand Counterclockwise",  # 22
        "Zooming In With Full Hand",  # 23
        "Zooming In With Two Fingers",  # 24
        "Zooming Out With Full Hand",  # 25
        "Zooming Out With Two Fingers"  # 26
    ]

    elif num_classes == 10:

        catigories = ["Fall", "SalsaSpin", "Taichi", "WallPushups", "WritingOnBoard", "Archery", "Hulahoop", "Nunchucks", "WalkingWithDog", "test"]

    elif num_classes == 3:

        catigories = ['Fall', "Not Fall", "Test"]


    return catigories


def main(num_classes):


    if num_classes not in [3, 10, 27]:
        return "Can only handle 2, 10, and 27 classes"

    else:
        catigories = get_categories(num_classes)

    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])


    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    torch_module = MobileNetV2(n_class=num_classes)


    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):  # checkpoint not downloaded
        print('Downloading PyTorch checkpoint...')
        url = 'https://hanlab.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    

    # PUT PATH FOR YOUR MODEL WHEN LODING model_new
    
    #model_new = torch.load("../../temporal-shift-module/pretrained/ckpt.best.pth.tar")
    #model_new = torch.load("../../pretrained/ckpt.best.pth.tar")

    model_new = torch.load("../../pretrained/2cat/ckpt.best.pth.tar")


    print(type(model_new['state_dict']))

    model_old = torch.load("mobilenetv2_jester_online.pth.tar")
    print(type(model_old))
    

    # Fixing new model parameter mis-match
    state_dict = model_new['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        #name = k[7:] # remove `module.`

        if "module.base_model." in k:
            name = k.replace("module.base_model.", "")

            if ".net" in name:
                name = name.replace(".net", "")


        elif "module." in k:
            name = k.replace("module.new_fc.", "classifier.")
        
        new_state_dict[name] = v

    # load params
    torch_module.load_state_dict(new_state_dict)    
    #torch_module.load_state_dict(model_old)

    torch_module.eval()

    #cap = cv2.VideoCapture(1)

    cap = cv2.VideoCapture('./zorian_0965.train.avi')

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    #img = '../../videos_rolling/000033.jpg'
    #img = Image.open(img)  

    
    # full_screen = False
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, 640, 480)
    # cv2.moveWindow(WINDOW_NAME, 0, 0)
    # cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)


    shift_buffer = [torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7])]


    print("HI")

    t = None    
    index = 0
    idx = 0
    history = [1]
    history_logit = []
    history_timing = []
    i_frame = -1

    #while True:
    success,img = cap.read()

    


    while success:
        
        i_frame += 1
        #_, img = cap.read()  # (480, 640, 3) 0 ~ 255


        if i_frame % 3 == 0:
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))

            prediction = torch_module(input_var, *shift_buffer)

            feat, shift_buffer = prediction[0], prediction[1:]


            if SOFTMAX_THRES > 0:

                print("here??")

                feat_np = feat.detach().numpy().reshape(-1)
                feat_np -= feat_np.max()

                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
        
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.detach().numpy(), axis=1)[0]
        
                else:
                    idx_ = idx
    
            else:
                coefs = feat.detach().numpy()[0]
                
                #print("coefs = " + str(coefs))
                ind = np.argpartition(coefs, -3)[-3:]                
                sorted_matches = ind[np.argsort(coefs[ind])] # these are reverse sorted, but not worth reversing 

                # print(sorted_matches)                

                # Top indx
                idx_ = sorted_matches[2]
                
                print(f"1. {catigories[idx_]}, {round(coefs[idx_],4)}")
                print(f"2. {catigories[sorted_matches[1]]}, {round(coefs[sorted_matches[1]],4)}")
                print(f"3. {catigories[sorted_matches[0]]}, {round(coefs[sorted_matches[0]],4)}")


            # Check history, average it out

            if HISTORY_LOGIT:
                history_logit.append(feat.detach().numpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]


            idx, history = process_output(idx_, history)
            

            t2 = time.time()
            print(f"Prediction @ Frame {index}: {catigories[idx]}")
            print("\n")

            
            current_time = t2 - t1

        if success:

            img = cv2.resize(img, (640, 480))
            img = img[:, ::-1]
            height, width, _ = img.shape
            label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

            # cv2.putText(label, 'Prediction: ' + catigories[idx], (0, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time), (width - 170, int(height / 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            img = np.concatenate((img, label), axis=0)
            #cv2.imshow(WINDOW_NAME, img)

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:  # exit
                break
            
            elif key == ord('F') or key == ord('f'):  # full screen
                print('Changing full screen option!')
                
                full_screen = not full_screen
                
                if full_screen:
                    print('Setting FS!!!')
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


        if t is None:
            t = time.time()
        
        else:
            nt = time.time()
            index += 1
            t = nt
        
        # Next frame
        success,img = cap.read()

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    print("Starting... \n")

    SOFTMAX_THRES = 0
    HISTORY_LOGIT = True
    REFINE_OUTPUT = True
    WINDOW_NAME = "GESTURE CAPTURE"

    #Modify number of classes here
    main(3)
    #main(10)

    print("Done")
