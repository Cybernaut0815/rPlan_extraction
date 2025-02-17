import os
import json
import random
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_descriptions(data, llm, system_message, query, description_number=3):
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=""),
    ]

    bedroom_synonyms = ["child room", "master room", "second room", "guest room", "study room"]


    def add_bedrooms(_jsonfile):
        jsonfile = deepcopy(_jsonfile)
        
        # Remove bedroom-type rooms from room_counts after counting
        for room in bedroom_synonyms:
            if room in jsonfile["room_counts"]:
                del jsonfile["room_counts"][room]
        
        # Update graph string - different approach for numbered suffixes
        graph_string = jsonfile["graph_string"]

        counter = 0
        
        for room in bedroom_synonyms:
            for i in range(10):
                old_name = f'"{room}_{i}"'
                if old_name in graph_string:
                    new_name = f'"bedroom_{counter}"'
                    graph_string = graph_string.replace(old_name, new_name)
                    counter += 1

        jsonfile["graph_string"] = graph_string
        jsonfile["room_counts"]["bedrooms"] = counter
        
        return jsonfile


    descriptions = []

    for i in range(description_number):
        if random.random() < 0.5:
            processed_data_1 = deepcopy(data)
            processed_data_1 = add_bedrooms(processed_data_1)
            # print(processed_data_1["room_counts"])
            # print(processed_data_1["graph_string"])
            messages[1] = HumanMessage(content=query + "\n" + "Room counts: " + json.dumps(processed_data_1["room_counts"]) + "\n" + "Room connections: " + json.dumps(processed_data_1["graph_string"]))
        else:
            processed_data_2 = deepcopy(data)
            # print(processed_data_2["room_counts"])
            # print(processed_data_2["graph_string"])
            messages[1] = HumanMessage(content=query + "\n" + "Room counts: " + json.dumps(processed_data_2["room_counts"]) + "\n" + "Room connections: " + json.dumps(processed_data_2["graph_string"]))
        
        result = llm.invoke(messages)
        descriptions.append(result.content)

    return descriptions

