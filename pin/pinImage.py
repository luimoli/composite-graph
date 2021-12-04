import os
import cv2
import numpy as np
import random
from utils.func import *

class PinImage:
    def __init__(self) -> None:
        pass

    def augmented(self, image, angle_range=(-5,5), scale_range=(0.5, 1.1)):
        """[rotate and scaling foreground-img for data augmentation: to generate different salient objects.]

        Args:
            image ([RGB]): [description]
            angle_range (tuple, optional): [get a random angle in this range]. Defaults to (-5,5).
            scale_range (tuple, optional): [get a random scaling factor in this range]. Defaults to (0.5, 1.1).

        Returns:
            [RGB]: [new img]
        """
        (h, w) = image.shape[:2]
        angle = random.randint(angle_range[0], angle_range[1])
        scale = random.uniform(scale_range[0], scale_range[1])
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h)) #h and w 's position
        return rotated

    def shape(self, pic):
        r,c = pic.shape[0:2]
        return r,c

    def resize(self, font, back):
        """[get the resized foreground-img adjusted to fit in the background.]
        Args:
            font ([RGB]): [foreground-img]
            back ([RGB]): [backgound-img]
        Returns:
            [type]: [description]
        """
        f_r,f_c = self.shape(font)
        b_r,b_c = self.shape(back)
        if b_c/f_c >=1 and b_r/f_r >=1:
            f_rs = font
        else:
            k = min(b_c/f_c, b_r/f_r)
            f_rs = cv2.resize(font,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
        return f_rs

    def get_fnew_and_alpha(self, f_rs):
        """[get alpha channel mask.]

        Args:
            f_rs ([type]): [description]

        Returns:
            [type]: [description]
        """
        b,g,r,a = cv2.split(f_rs)
        f_new = cv2.merge((b,g,r))
        f_new = f_new.astype(float)
        alpha_255 = cv2.merge((a,a,a))
        alpha_ori = alpha_255.copy()
        index = np.argwhere(alpha_255 > 0)
        for i in index:
            alpha_255[i[0]][i[1]][i[2]] = 255
        return f_new, alpha_255, alpha_ori


    def get_bnew(self, f_rs,back,a,b,c,d,alpha_path):
        """[generate composite-graph using cordinates.]
        Args:
            f_rs ([type]): [description]
            back ([type]): [description]
            a ([type]): [description]
            b ([type]): [description]
            c ([type]): [description]
            d ([type]): [description]
            alpha_path ([type]): [description]
        Returns:
            [type]: [description]
        """
        b_new = back[a:b,c:d].copy()
        b_new = b_new.astype(float)

        alpha_full = np.zeros(back.shape, dtype=back.dtype)
        #alpha_full = cv2.imread(old_a_pth)#读入上一张完整图片的alpha图
        af = alpha_full[a:b,c:d].copy()
        alpha_full[a:b,c:d] = cv2.add(af, self.get_fnew_and_alpha(f_rs)[1])
        cv2.imwrite(alpha_path, alpha_full)

        return b_new

    def conver(self, f_rs,b_new):
        """[merge the foreground and backgound.]
        Args:
            f_rs ([type]): [description]
            b_new ([type]): [description]
        Returns:
            [RGB]: [merged-img with size of foreground-img.]
        """
        f_new,alpha_255,alpha_ori = self.get_fnew_and_alpha(f_rs)
        alpha = alpha_ori.astype(float) / 255
        f_new = cv2.multiply(alpha,f_new)
        b_new = cv2.multiply(1-alpha,b_new)
        outImage = f_new + b_new
        return outImage

    def choose_pos_center(self, fp,bp):
        print('this is center!')
        r, c = self.shape(fp)
        br,bc = self.shape(bp)
        r_ = (br - r)//2
        c_ = (bc - c)//2
        size = [r_, r_+r, c_, c_+c ]
        return size

    def choose_pos_random(self, fp,bp):
        print('this is random!')
        r1, c1 = self.shape(fp)
        b_r,b_c = self.shape(bp)
        xr = random.randint(0, b_r-r1)
        xc = random.randint(0,b_c-c1)
        size = [xr,xr+r1,xc,xc+c1]
        return size

    def pin(self, font, back, alpha_savepath, res_savepath, method):
        """[pin the merged-img on bg-img]

        Args:
            font ([RGB]): [description]
            back ([RGB]): [description]
            alpha_savepath ([str]): [description]
            res_savepath ([str]): [description]
            method ([str]): [description]
        """
        f_rs = self.resize(font,back)
        if method == 'center':
            a,b,c,d = self.choose_pos_center(f_rs,back)
        elif method == 'random':
            a,b,c,d = self.choose_pos_random(f_rs,back)
        else:
            print('wrong pin method! choose between center and random!')
        b_new = self.get_bnew(f_rs,back,a,b,c,d,alpha_savepath)
        outImage = self.conver(f_rs,b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        cv2.imwrite(res_savepath, res)


    def batch(self, src_path, dst_path, res_path, gt_path, center_num, augmented_flag, bg_repeat_flag):
        """[summary]

        Args:
            src_path ([str]): [foreground-imgs' path]
            dst_path ([str]): [background-imgs' path]
            res_path ([str]): [set a path to save cg-img]
            gt_path ([str]): [set a path to save gt of cg-img]
            center_num ([int]): [the number of cg-img which has centered salient object.]
            augmented_flag ([bool]): [whether to ]
            bg_repeat_flag ([type]): [description]
        
        result cg-img/cg-img-gt 's naming format: foreground_name + backgound_name . png/jpg
        """
        bg_path_list = [] # full-paths of bg-imgs
        bg_name_list= []  # names og bg-imgs
        for dst_item in os.listdir(dst_path):
            back_path = os.path.join(dst_path, dst_item)
            bg_path_list.append(back_path)
            bg_name_list.append(dst_item[:-4])

        total = len(bg_path_list)
        pos, tmp = 0, 0
        for img_item in os.listdir(src_path):
            # choose font image---------------------------------------
            img_path = os.path.join(src_path, img_item)
            img = cv2.imread(img_path,-1)
            src_new = self.augmented(img) if augmented_flag else img

            # choose back image----------------------------------------
            if bg_repeat_flag:
                l_num = random.randint(0, total-1) # choose bg-img randomly
                dst_new = cv2.imread(bg_path_list[l_num],-1)
                bg_name = bg_name_list[l_num]
            else:
                dst_new = cv2.imread(bg_path_list[pos],-1)
                bg_name = bg_name_list[pos]
                pos +=1

            # set all the write paths-----------------------------
            res_set_name = img_item[:-4]+'_'+ bg_name
            gt_path_set = os.path.join(gt_path, res_set_name +'.png') 
            resimg_path_set = os.path.join(res_path, res_set_name +'.jpg')
            # dic_set = os.path.join(dict_path, res_set_name +'.txt')
            # ins_set = os.path.join(color_path, res_set_name +'.png')

            method = 'center' if tmp < center_num else 'random'
            self.pin(src_new, dst_new, gt_path_set, resimg_path_set, method)
            tmp += 1




