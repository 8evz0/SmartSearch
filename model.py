
import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image
import hashlib
from pathlib import Path
from typing import Callable
import json
import random


# Функции, которые предоставляет API этого бекенда:
# initialize_values - загружает необходимые глобальные переменные, выполнить в начале программы
# fit_canvas - вписывает PIL.Image в холст выбранного размера - по центру на прозрачном фоне, это будет полезно при создании основного табличного элемента с выводимыми картинками
# generate_thumbs - генерирует эти превью для всех картинок в папке (рекурсивно) и помещает их в папку
# ml_import, ml_model_import - импорт библиотек и модели, занимает много времени, потому вынесен в отдельные функции
# analyze_image - возвращает уникальные метки объектов, которе модель нашла на изображении
# labelcache_write - записать метки в json файл
# generate_labels - применяет модель для всех картинок в папке (рекурсивно), записывает результат в глобальную переменную

# Файлы, которые создаёт программа:
# файл с настройками
# файл с кэшированными метками
# папка со сгенерированными превью изображений


# прога ищет файл settings.json, откуда подгружаются значения этих переменных:
# если не нашла, то значения берутся дефолтные
APP_SETTINGS = {
    'IMG_FILETYPES':['.jpg','.png'], # поддерживаемые типы изображений, надо добавить больше, но нужно тестирование
    'THUMBDIR_NAME':'thumbs', # имя папки, в которой хранятся превью изображений
    'LABEL_CACHE_NAME':'labelCache', # имя файла, куда кэшируются присвоенные метки
    'ROOTPATH':None # корневая папка работы программы, там будут хранится кэш и папка с превью, но не файл настроек
}


def initialize_values():
    """Run this first to create all the necessary global variables\n
    Creates global variables:\n
    all_labels - dictionary of ID : labels with all of the analyzed images\n
    all_thumbs - dictionary of ID : path with all of the created thumbs\n
    IMG_FILETYPES - supported image formats\n
    THUMBDIR_NAME - name of the thumbnails directory\n
    LABEL_CACHE_NAME - name of the labelCache file\n
    ROOTPATH - root folder of the program folder, contains everything except for the settings file\n
    ML_IMPORT_DONE - whether ml stuff was already imported, doesn't apply outside jupyter use\n\n
    Global variable dependencies:\n
    APP_SETTINGS - default values for settings"""
    global all_labels,all_thumbs,APP_SETTINGS
    all_labels = {}
    all_thumbs = {}

    # обновляем настройки, если был найден файл
    if Path.is_file(Path('./settings.json')):
        APP_SETTINGS = json.load(Path.open(Path('./settings.json'),'r'))
    else:
        with Path.open(Path('./settings.json'),'w') as f:
            json.dump(APP_SETTINGS,f)

    global IMG_FILETYPES,THUMBDIR_NAME,LABEL_CACHE_NAME,ROOTPATH,ML_IMPORT_DONE
    IMG_FILETYPES = APP_SETTINGS['IMG_FILETYPES']
    THUMBDIR_NAME = APP_SETTINGS['THUMBDIR_NAME']
    LABEL_CACHE_NAME = APP_SETTINGS['LABEL_CACHE_NAME']
    ROOTPATH = APP_SETTINGS['ROOTPATH']
    ML_IMPORT_DONE = False

    # заплатка для работы в блокноте
    if not ROOTPATH:
        if '__file__' not in globals():
            ROOTPATH = Path('.').absolute()
        else:
            ROOTPATH = Path(__file__).parent.absolute()

    # загружаем уже известные метки, если был найден файл
    if Path.is_file(ROOTPATH/(LABEL_CACHE_NAME+'.json')):
        all_labels = json.load((ROOTPATH/(LABEL_CACHE_NAME+'.json')).open())

    # загружаем все уже созданные превью изображений
    if Path.is_dir(ROOTPATH/THUMBDIR_NAME):
        all_thumbs = {x.stem:x for x in Path.glob(ROOTPATH/THUMBDIR_NAME,'**/*') if x.suffix in IMG_FILETYPES}


# по умолчанию ID изображения - md5 хэш файла
def md5(fname: str | Path) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def fit_canvas(image_path: Path, canvas_size: tuple[int,int]) -> Image.Image:
    """Fits image in a canvas of given size (places it in a center of a transparent canvas, resized to match either width or height of canvas)\n
    Parameters:\n
    image_path - (absolute) path to a given image\n
    canvas_size - size of a canvas to fit that image into"""
    im = Image.open(image_path)
    im.thumbnail(canvas_size)
    img_canvas=Image.new('RGBA',canvas_size,(0,0,0,0))
    center_spot: tuple[int,int] = ((canvas_size[0]-im.size[0])//2,(canvas_size[1]-im.size[1])//2)
    img_canvas.paste(im,center_spot)
    return img_canvas


def generate_thumbs(thumb_folder_path: Path, img_foler_path: Path, canvas_size: tuple[int,int], namefunc: Callable[[Path],str] = md5) -> None:
    """Recursively searches for images in a given folder, creating canvasThumbnail for each of them in a root_path/THUMBDIR_NAME\n
    Parameters:\n
    thumb_folder_path - absotule path to thumb folder (would be created if doesnt exist)\n
    img_folder_path - absolute path to the image search root folder\n
    canvas_size - size of a canvas to fit that image into\n
    namefunc - function, which creates a name of a thumbnail image (default is md5 hash) without extension\n
    Creates global variables:\n
    all_thumbs - dictionary of ID : path with all of the created thumbs"""
    global all_thumbs
    if not Path.is_dir(thumb_folder_path): Path.mkdir(thumb_folder_path)
    for img_path in Path.glob(img_foler_path,'**/*'):
        if img_path.suffix not in IMG_FILETYPES: continue
        thumb_name = namefunc(img_path) + '.png'
        if not Path.is_file(thumb_folder_path / thumb_name):
            fit_canvas(img_path,canvas_size).save(thumb_folder_path / thumb_name) 
    all_thumbs = {x.stem:x for x in Path.glob(ROOTPATH/THUMBDIR_NAME,'**/*') if x.suffix in IMG_FILETYPES}


def ml_import():
    """Imports ml modules.\n
    Creates global variables:\n
    torch - torch module\n
    torchvision - torchvision module"""
    global torchvision
    global torch
    import torchvision
    import torch
    torch.set_grad_enabled(False)


def ml_model_import():
    """Imports ml pretrained model.\n
    Global variable dependencies:\n
    torchvision - torchvision module\n
    Creates global variables:\n
    model - torchvision model"""
    global model
    global torchvision
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model = model.eval().cpu()


def analyze_image(imgpath: Path) -> list[str]:
    """Returns unique labels of all of the detected objects in a given image according to model results.\n
    Global variable dependencies:\n
    model - torchvision model"""
    coco_names = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    global model
    image = Image.open(imgpath).convert('RGB')
    image_tensor = torchvision.transforms.functional.to_tensor(image).cpu()
    output = model([image_tensor])[0]

    labls = set()

    for i in range(len(output['scores'])):
        if output['scores'][i] <= 0.5:
            continue
        labls = labls.union([int(output['labels'][i].cpu().numpy())])
    found_labels = [coco_names[x] for x in labls]

    return found_labels


def labelcache_write():
    """Write all known labels to json file.\n 
    Global variable dependencies:\n
    all_labels - dict of ID : labels"""
    global all_labels
    with Path.open(ROOTPATH/(LABEL_CACHE_NAME+'.json'),'w') as f:
        json.dump(all_labels,f)


def count_images(img_foler_path: Path) -> int:
    """Returns number of images found recursively, starting from folder"""
    return len([x for x in Path.glob(img_foler_path,'**/*') if x.suffix in IMG_FILETYPES])


def generate_labels(img_foler_path: Path, idfunc: Callable[[Path],str] = md5, tag: str | None = None) -> None:
    """Recursively searches for images in a given folder, applying model to each of them\n
    Parameters:\n
    img_folder_path - absolute path to the image search root folder\n
    idfunc - function, which creates an ID of an image (default is md5 hash, must be the same as for the thumbs)\n
    tag - optional tag of DearPyGui element - the function would update that element's value according to analysis progress\n
    Global variable dependencies:\n
    all_labels - dict of ID : list of labels"""
    global all_labels
    img_num = count_images(img_foler_path)
    curr_img_num = 1
    for img_path in Path.glob(img_foler_path,'**/*'):
        if img_path.suffix not in IMG_FILETYPES: continue
        imgid = idfunc(img_path)
        if imgid not in all_labels.keys(): 
            all_labels[imgid] = analyze_image(img_path)
        #print(str(curr_img_num)+'/'+str(img_num)+':', img_path,imgid,all_labels[imgid])
        if tag: dpg.set_value(tag,f"Analysis progress: {curr_img_num}/{img_num}")
        curr_img_num+=1

def search_image(lable:str)->list:
    """Search for all thumbs in all_thumbs that marked with lable\n
    Parameters:\n
    lable - lable to search for
    Global variable dependencies:
    all_labels - dict of ID : list of labels
    all_thumbs - dictionary of ID : path with all of the created thumbs
    """
    global all_labels, all_thumbs
    matched_images = []
    for key in all_thumbs.keys():
        if lable in all_labels[key]:
            matched_images.append(key)
    return matched_images



# дальше идёт код, связанный с демонстрационной гуи. Без комменариев



def first_callback(sender, app_data):
    dpg.set_value("imgfolderpath", app_data['file_path_name'])
    img_folder_path = app_data['file_path_name']
    img_folder_path = Path(img_folder_path)
    if not Path.is_dir(img_folder_path):
        dpg.set_value('status','error happened')
    else:
        img_count = count_images(img_folder_path)
        dpg.configure_item('countimages',enabled=False)
        dpg.set_value('status','generating thumbs')
        generate_thumbs(ROOTPATH/THUMBDIR_NAME,img_folder_path,(256,256))
        dpg.set_value('status',str(img_count)+' images found')
        dpg.configure_item('countimages',enabled=True)

def cancel_folder_selection(sender, app_data):
    pass

def second_callback(sender, app_data):
    global ML_IMPORT_DONE
    dpg.configure_item(sender,enabled=False)
    if not ML_IMPORT_DONE:
        dpg.set_value('status','importing ml libraries')
        ml_import()
        dpg.set_value('status','importing ml model')
        ml_model_import()
        ML_IMPORT_DONE = True
    img_folder_path = dpg.get_value('imgfolderpath')
    img_folder_path = Path(img_folder_path)
    generate_labels(img_folder_path,tag='status')
    generate_thumbs(ROOTPATH/THUMBDIR_NAME,img_folder_path,(256,256))
    labelcache_write()
    dpg.set_value('status','Analysis complete, results saved')
    dpg.configure_item(sender,enabled=True)
    

def third_callback(sender, app_data):
    global all_thumbs,all_labels
    for index, _ in enumerate(all_thumbs.keys()):
        if dpg.does_alias_exist(f"texture-{index}"):
            pass
        else:
            dpg.add_dynamic_texture(width=256, height=256, default_value=[0 for _ in range(256*256*4)], tag=f"texture-{index}", parent="texture_reg") 
        
    for index, key in  enumerate(all_thumbs.keys()):
        img_path = all_thumbs[key]
        img_arr = np.true_divide(np.array(Image.open(img_path).convert('RGBA')),255.0)

        dpg.set_value(f"texture-{index}", img_arr)
        dpg.add_image(f"texture-{index}", indent=0, parent="img_show")
        if dpg.does_alias_exist(f"img-text-{index}"):
            pass
        else:
            dpg.add_text(" ".join(all_labels[key]),parent="img_show", tag=f"img-text-{index}")
    dpg.configure_item("filter_input", enabled=True)
        

    

def filter_callback(sender, filter_string):
    dpg.set_value("filter_id", filter_string)
    img_list = search_image(filter_string)
    if (img_list):
        for index, key in  enumerate(all_thumbs.keys()):
            dpg.set_value(f"texture-{index}", [0 for _ in range(256*256*4)])
            dpg.set_value(f"img-text-{index}", "")
        for index, key in  enumerate(img_list):
            img_path = all_thumbs[key]
            img_arr = np.true_divide(np.array(Image.open(img_path).convert('RGBA')),255.0)

            dpg.set_value(f"texture-{index}", img_arr)
            dpg.set_value(f"img-text-{index}", " ".join(all_labels[key]))
    else:
        for index, key in  enumerate(all_thumbs.keys()):
            img_path = all_thumbs[key]
            img_arr = np.true_divide(np.array(Image.open(img_path).convert('RGBA')),255.0)

            dpg.set_value(f"texture-{index}", img_arr)
            dpg.set_value(f"img-text-{index}", " ".join(all_labels[key]))
    



    
CLASSES = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def main():
    
    initialize_values()

    dpg.create_context()

    with dpg.theme() as disabled_theme:
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [29, 29, 29])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 29, 29])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [29, 29, 29])
    dpg.bind_theme(disabled_theme)


    dpg.add_file_dialog(directory_selector=True, show=False, callback=first_callback, tag="file_dialog_id", width=700 ,height=400, cancel_callback=cancel_folder_selection)
    dpg.add_texture_registry(tag="texture_reg", show=False)
    dpg.create_viewport(title='test', width=1280, height=720)



    with dpg.window(label="Tutorial",tag='main'):
        with dpg.group(horizontal=True):
            dpg.add_text(tag='imgfolderpath',default_value='./path_to_images')
            dpg.add_button(label="Set Directory", tag='countimages', callback=lambda: dpg.show_item("file_dialog_id"))

        dpg.add_button(label='Analyze images',tag='analyzeimages',callback=second_callback)

        dpg.add_input_text(label="Available classes", callback=filter_callback, width=221, enabled=False, tag="filter_input")
        
        with dpg.group(horizontal=False):
            with dpg.child_window(height=100, width=256, horizontal_scrollbar=False):
                dpg.add_text("All classes")
                with dpg.filter_set(id="filter_id"):
                    for i in CLASSES:
                        dpg.add_text(f"{i}", filter_key=f"{i}", bullet=True)
                        
        dpg.add_button(label='Show images',tag='showimage',callback=third_callback)
        with dpg.group(horizontal=False):
            with dpg.child_window(height=-100, width=-1, tag="img_show"):
                pass

        dpg.add_text(tag='status',default_value='status would be displayed here',wrap=256)

    dpg.set_primary_window('main',True)
    dpg.show_item_registry()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__=='__main__': main()


