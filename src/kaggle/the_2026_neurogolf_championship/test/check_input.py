import json
import numpy as np

from pathlib import Path

dir_path = Path("neurogolf-2026")

# Reads only text files in the top-level directory
for file_path in dir_path.glob("*.json"):
    #print('-'*50)
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if(len(data["test"])!=1):
        print(file_path)
    '''for i, pair in enumerate(data["train"][:2]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        print(inp.shape)
        print(out.shape)
        if(inp.shape == out.shape):
            print(i)
    for i, pair in enumerate(data["arc-gen"][:2]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        print(inp.shape)
        print(out.shape)'''