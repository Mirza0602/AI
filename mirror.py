from PIL import Image
from PIL import ImageOps
from pathlib import Path


def mirror_images(directory):
    print(directory)
    for path in Path(directory).glob("*.png"):
        im = Image.open(path)
        im = ImageOps.mirror(im)
        im.save(path)


mirror_images("./digits_mirror/train/2")
mirror_images("./digits_mirror/train/3")
mirror_images("./digits_mirror/train/5")
mirror_images("./digits_mirror/test/2")
mirror_images("./digits_mirror/test/3")
mirror_images("./digits_mirror/test/5")