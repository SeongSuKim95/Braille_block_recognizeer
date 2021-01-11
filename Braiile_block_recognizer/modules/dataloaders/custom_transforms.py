import torch
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
class Normalize(object):

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomcolorJitter(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        transform = transforms.ColorJitter(brightness=(0.5,0.7))

        if random.random() <0.3:
            img = transform(img)
        return {'image': img,
                'label': mask}
# img = transforms.ToPILImage()(torch.randn(3, 224, 224))
# crop = transforms.RandomResizedCrop(224)
# params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
# img_crop = transforms.functional.crop(img, *params)


class test(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = np.random.randint(2147483647)
        transform = transforms.RandomPerspective(distortion_scale=0.5, p=1)

        if random.random() <1:
            random.seed(seed)
            img = transform(img)
            random.seed(seed)
            mask= transform(mask)
        return {'image': img,
                'label': mask}
# img = transforms.ToPILImage()(torch.randn(3, 224, 224))
# crop = transforms.RandomResizedCrop(224)
# params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
# img_crop = transforms.functional.crop(img, *params)

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, height, width):
        self.size = (width, height)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img,
                'label': mask}

class RandomShadow(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img,
                'label': mask}

def generate_shadow_coordinates(imshape, no_of_shadows):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(3,15)):

            # Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
            vertices_list.append(vertices)

            return vertices_list ## List of shadow vertices

def add_shadow(image,no_of_shadows):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows)
    #3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 0) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5
        ## if red channel is hot, image's "Lightness" channel's brightness is lowered
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return image_RGB

if __name__ == '__main__':
    image = cv2.imread('MP_SEL_SUR_000004.jpg',cv2.IMREAD_COLOR)
    shadow= add_shadow(image,3)
    cv2.imshow('shadow',shadow)
    cv2.waitKey(0)
