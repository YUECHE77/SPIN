import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset

from utils import load_image


class COCODataSet(Dataset):
    def __init__(self, data_path, trans):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        img_path = os.path.join(self.data_path, img_file)
        image = Image.open(img_path).convert("RGB")
        image = self.trans(image)

        return {"img_id": img_id, "image": image, "image_path": img_path}


class POPEChatDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []

        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == "no":
                    label_list[i][j] = 0
                else:
                    label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")

        image = self.trans(raw_image)

        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label, "image_path": image_path}


class MMHalDataset(Dataset):
    def __init__(self, json_path, trans):
        self.json_path = json_path
        self.json_data = json.load(open(self.json_path, 'r'))

        self.trans = trans

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        line = self.json_data[index]
        question = line['question']
        img_src = line['image_src']
        image = load_image(img_src)
        image = self.trans(image)

        return {'question': question, 'image': image, 'image_path': img_src}
