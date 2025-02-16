import os
import json
import random

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_descriptions(data, llm, system_message, query, description_number=3):
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query + "\n" + json.dumps(data["room_counts"])),
    ]

    bedroom_synonyms = ["child room", "master room", "second room", "guest room", "study room"]

    def add_bedrooms(jsonfile):
        jsonfile = jsonfile.copy()
        bedrooms = 0
        for room in jsonfile:
            if room in bedroom_synonyms:
                bedrooms += jsonfile[room]
        jsonfile["bedrooms"] = bedrooms
        
        # Remove bedroom-type rooms after counting
        for room in bedroom_synonyms:
            if room in jsonfile:
                del jsonfile[room]

        return jsonfile
    
    def remove_empty_rooms(jsonfile):
        jsonfile = jsonfile.copy()
        # Create list of rooms to remove to avoid modifying dict during iteration
        rooms_to_remove = [room for room in jsonfile if jsonfile[room] == 0]
        for room in rooms_to_remove:
            del jsonfile[room]
        return jsonfile


    descriptions = []

    for i in range(description_number):

        if random.random() < 0.5:
            room_data = add_bedrooms(data["room_counts"])
            room_data = remove_empty_rooms(room_data)
            messages[1] = HumanMessage(content=query + "\n" + json.dumps(room_data))
        else:
            room_data = remove_empty_rooms(data["room_counts"])
            messages[1] = HumanMessage(content=query + "\n" + json.dumps(room_data))
        
        result = llm.invoke(messages)
        descriptions.append(result.content)

    return descriptions
