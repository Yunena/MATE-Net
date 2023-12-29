import torch
import torchio as tio


TARGET_SIZE = (256,256,32)

def random_flip(file):
    image = tio.ScalarImage(file)
    flip = tio.RandomFlip(axes= 'LR', flip_probability=0.5)
    return flip(image)

def random_zoom(file):
    image = tio.ScalarImage(file)
    zoom_ = tio.RandomAffine(0.5,0,0)
    return zoom_(image)

def random_rotate(file):
    image = tio.ScalarImage(file)
    rotate_ = tio.RandomAffine(0,45,0)
    return rotate_(image)

def random_translate(file):
    image = tio.ScalarImage(file)
    translate_ = tio.RandomAffine(0,0,20)
    return translate_(image)

def random_affine(file):
    image = tio.ScalarImage(file)
    affine_ = tio.RandomAffine(0.5,45,20)
    return affine_(image)


def zoom(file,scales):
    image = tio.ScalarImage(file)
    zoom_ = tio.Affine(scales,0,0)
    return zoom_(image)

def rotate(file,degree):
    image = tio.ScalarImage(file)
    rotate_ = tio.Affine(1,degree,0)
    return rotate_(image)

def translate(file,trans):
    image = tio.ScalarImage(file)
    translate_ = tio.Affine(1,0,trans)
    return translate_(image)

def affine(file,scales,degree,trans):
    image = tio.ScalarImage(file)
    croptrans = tio.CropOrPad(TARGET_SIZE)
    image = croptrans(image)
    affine_ = tio.Affine(scales,degree,trans)
    return affine_(image)

def composetrans(file,scales,degree,trans,target_size=(256,256,32)):
    image = file
    translist = [
        tio.CropOrPad(target_size),\
        tio.RandomFlip(axes= 'LR', flip_probability=0.5), \
        tio.Affine(scales, degree, trans)
    ]
    composetrans_ = tio.Compose(translist)
    return composetrans_(image)

