import random
import json
import torch
from .model import neuralNet
from .bott import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('D:/ddjango/microweb/microwebb/blog/intents.json') as f:
    intents= json.load(f)

FILE= "D:/ddjango/microweb/microwebb/blog/data.pth"
data=torch.load(FILE)
inputsize= data["input_size"]
hiddensize=data["hidden_size"]
outputsize=data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model =neuralNet(inputsize, hiddensize, outputsize).to(device)

model.load_state_dict(model_state)
model.eval()


bot_name="Jarvis"
print("Lets chat!!!!! type quit to quit")




def jarvischat(sent):
    # sent = input("You: ")

    # if sent =="quit":
    #     break

    sent= tokenize(sent)
    X=bag_of_words(sent, all_words)
    X= X.reshape(1,X.shape[0])
    X= torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{bot_name}: {random.choice(intent['responses'])}"

    else:
        return f"{bot_name}: I don't understand....."



# while True:
#     sent = input("You: ")

#     if sent =="quit":
#         break

#     sent= tokenize(sent)
#     X=bag_of_words(sent, all_words)
#     X= X.reshape(1,X.shape[0])
#     X= torch.from_numpy(X)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)
#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]

#     if prob.item() > 0.75:
#         for intent in intents["intents"]:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")

#     else:
#         print(f"{bot_name}: I don't understand.....")

