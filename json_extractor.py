# %%

import json
import os
from PIL import Image
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from helpers.utils import resize_plan
from helpers.llm_utils import get_descriptions

from tqdm import tqdm

# %%

DATA_PATH = r"G:\Datasets\rPlan\dataset\dataset\floorplan_dataset"
OUT_PATH = r"G:\Datasets\rPlan\dataset\dataset\floorplan_dataset_processed_16x16"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

#%%

# Get list of files in directory and sort them by the number in the filename
files = sorted(os.listdir(DATA_PATH), key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[-1].split('.')[0]))))


# Find first image file
image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
if image_files:
    first_image_path = os.path.join(DATA_PATH, image_files[5])
    
    # Open the first image
    img = Image.open(first_image_path)
else:
    print("No image files found in directory")


# %%

arrayImg = np.array(img)
print(arrayImg.shape)

# %%

room_type_channel = arrayImg[:,:,1]

plt.imshow(room_type_channel, cmap='viridis')
plt.show()

# %%

point_count = 32

# Create a 16x16 grid of points
x = np.linspace(0, arrayImg.shape[1], point_count+1)
y = np.linspace(0, arrayImg.shape[0], point_count+1)
X, Y = np.meshgrid(x, y)

# Remove outer points by slicing the meshgrid arrays
X = X[:-1, :-1]
Y = Y[:-1, :-1]

# Calculate spacing between points
x_spacing = x[1] - x[0]
y_spacing = y[1] - y[0]

# Shift grid by half the spacing
X = X + x_spacing/2
Y = Y + y_spacing/2



# Plot the image
plt.imshow(room_type_channel, cmap='viridis')

# Overlay the grid points
plt.plot(X, Y, 'r.', markersize=5)
plt.show()


# %%

#processed_values = process_grid_points(room_type_channel, X, Y)
processed_values = resize_plan(arrayImg, X, Y)

#Show as image plot
plt.figure(figsize=(10,10))
plt.imshow(processed_values[:,:,1], cmap='viridis')
plt.colorbar()
plt.show()



# %%

# Reshape processed_values to be 16x16x3
resized_values = cv2.resize(processed_values, (16, 16), interpolation=cv2.INTER_NEAREST)

# Show resized values
plt.figure(figsize=(10,10))
plt.imshow(resized_values[:,:,0], cmap='viridis')
plt.colorbar()
plt.title('Resized to 16x16')
plt.show()


# %% 

funtions_dict = {"living room": 0, 
                "master room": 1, 
                "kitchen": 2, 
                "bathroom": 3, 
                "dining room": 4, 
                "child room": 5,
                "study room": 6,
                "second room": 7,
                "guest room": 8,
                "balcony": 9,
                "entrance": 10,
                "storage": 11}

# %%

def get_room_types_count(data, funtions):
    count_dict = {}
    for key, value in funtions.items():
        room_values = data[:,:,0]
        room_values = np.where(room_values == value)
        
        # Get the values from channel 1 where room_type matches
        masked_values = data[:,:,1][room_values]
        # Get unique values in the masked region
        unique_values = np.unique(masked_values)
        # Count number of unique values
        count_dict[key] = len(unique_values)
    return count_dict

# %%
from pprint import pprint

room_types_count = get_room_types_count(resized_values, funtions_dict)
pprint(room_types_count)

# %%

# # Show mask for balcony (value 9)
# plt.figure(figsize=(10,10))
# plt.imshow(arrayImg[:,:,1] == 9, cmap='binary')
# plt.colorbar()
# plt.title('Mask for Balcony')
# plt.show()

# %%

import json
import numpy as np

# Create dictionary to hold all data
output_data = {
    "room_counts": room_types_count,
    "dimensions": [resized_values.shape[0], resized_values.shape[1]],
    "function_values": resized_values[:,:,0].tolist(),
    "mask": resized_values[:,:,2].tolist()
}

# # Save to JSON file
# with open('room_data.json', 'w') as f:
#     json.dump(output_data, f, indent=4)

# %%

pprint(output_data)

# %%

# llm stuff after here

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
print(f"OPENAI_API_KEY found in .env: {os.getenv('OPENAI_API_KEY') != None}")


#%%

# load the llm

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.)

system_message = "You are a helpful assistant that creates a text description of a floor plan based on the given JSON data containing information about the room types and their counts. Be concise and to the point, not too long or too poetic. Don't try to sell the flat and don't explain what the rooms are for."
query = "Write a short descriptions like a prompt for a floorplan with the features. The prompt should be in natural language and describe the the flat based on the given count of different room types."

messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=query + "\n" + json.dumps(output_data["room_counts"])),
]

# %%

# get the descriptions
descriptions = get_descriptions(output_data, llm, system_message, query)
pprint(descriptions)


# %%


selected_files = files[:10]

for file in tqdm(selected_files):
    img = Image.open(os.path.join(DATA_PATH, file))
    arrayImg = np.array(img)
    processed_values = resize_plan(arrayImg, X, Y)
    resized_values = cv2.resize(processed_values, (16, 16), interpolation=cv2.INTER_NEAREST)
    
    #save the resized values
    cv2.imwrite(os.path.join(OUT_PATH, file), resized_values)

    room_types_count = get_room_types_count(resized_values, funtions_dict)


    data = {
        "room_counts": room_types_count,
        "dimensions": [resized_values.shape[0], resized_values.shape[1]],
        "functions": resized_values[:,:,0].tolist(),
        "mask": resized_values[:,:,2].tolist()
    }

    descriptions = get_descriptions(data, llm, system_message, query)
    data["descriptions"] = descriptions

    #save the data
    with open(os.path.join(OUT_PATH, file.split('.')[0] + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


