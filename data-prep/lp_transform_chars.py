import json
from shutil import copyfile
import random
from itertools import combinations
import os

def folderChoice():
    rand = random.randrange(10)
    if rand < 1:
        return "validate"
    elif rand <= 3:
        return "test"
    else: return "train"

if __name__ == "__main__":
    with open("data/resized_char_and_colour_0_3000_filtered.json") as json_file:
        split = False # change to true for validate, test, train split
        data = json.load(json_file)
        characters = [1,3,5,6] # desired characters
        combination_list = [combinations([1,3,5,6], i) for i in range(1,len(characters) + 1)]
        print(combination_list)
        character_combs = {}
        character_combs[15] = []
        id = 0
        for combs in combination_list:
            for comb in list(combs):
                character_combs[id] = list(comb)
                id += 1
        counts = {key: 0 for key in character_combs.keys()}
        for entry in data:
            path = "data/multi_char"
            curr_chars = data[entry]["Characters"]
            main_chars = [int(x) for x in curr_chars if int(x) in characters]
            for key in character_combs.keys():
                if set(character_combs[key]).issubset(set(main_chars)) and set(main_chars).issubset(set(character_combs[key])):
                    path = os.path.join(path, str(key))
                    counts[key] += 1

            if not os.path.exists(path):
                os.mkdir(path)
            if split:
                os.path.join(folderChoice(), entry)

            if set([int(x) for x in curr_chars]).issubset(set(characters)):
                copyfile(os.path.join("data/resized/resized", entry), os.path.join(path, entry))





