import pysam
import json
import random

samfile = pysam.AlignmentFile("SM47_IVT_NM_RNA004.IVT-i.filtered.bam", "rb")

group0 = ["IVT-i-2", "IVT-i-4"]
group1 = ["IVT-i-1", "IVT-i-3"]

#Split bam file into two groups
transcripts0 = []
transcripts1 = []
grouptoreadID = dict()
count1, count2, count3, count4 = 0,0,0,0
minCount = 400265
for line in samfile: #Cut all entries past the 400265th
    transcript = line.reference_name
    readID = line.query_name
    match transcript:
        case "IVT-i-1": 
            if count1 >= 400265:
                continue
            count1 += 1
        case "IVT-i-2":
            if count2 >= 400265:
                continue
            count2 += 1
        case "IVT-i-3":
            if count3 >= 400265:
                continue
            count3 += 1
        case "IVT-i-4":
            if count4 >= 400265:
                continue
            count4 += 1
        case _:
            raise Exception
    
    if transcript in group0:
        transcripts0.append((readID, transcript))
    elif transcript in group1:
        transcripts1.append((readID, transcript))

#Shuffle each group
random.shuffle(transcripts0) 
random.shuffle(transcripts1)

def split_data(train: float, test: float, values: list, name: str):
    '''
    Splits a list into train/test/val and stores into files "{name}Train.txt",
    "{name}Test.txt", or "{name}Validation.txt"

    args:
    train: relative size of train set (0 to 1)
    test: relative size of test set (0 to 1)
    values: list of values
    name: file name
    '''
    n = len(values)
    n1 = int(n*train)
    n2 = n1 + int(n*test) 

    training = values[:n1]
    test = values[n1:n2]
    validation = values[n2:]
    print("\n"+name+"\n")
    with open(f"./data/{name}Train.txt", "w") as outfile:
        count1, count2, count3, count4 = 0,0,0,0
        for rid in training:
            outfile.writelines(rid[0]+"\n")
            match rid[1]:
                case "IVT-i-1": count1+=1
                case "IVT-i-2": count2+=1
                case "IVT-i-3": count3+=1
                case "IVT-i-4": count4+=1
        print(f"Train dataset:\nIVT-i-1: {count1}\nIVT-i-2: {count2}\nIVT-i-3: {count3}\nIVT-i-4: {count4}")
    with open(f"./data/{name}Test.txt", "w") as outfile:
        count1, count2, count3, count4 = 0,0,0,0
        for rid in test:
            outfile.writelines(rid[0]+"\n")
            match rid[1]:
                case "IVT-i-1": count1+=1
                case "IVT-i-2": count2+=1
                case "IVT-i-3": count3+=1
                case "IVT-i-4": count4+=1
        print(f"Test dataset:\nIVT-i-1: {count1}\nIVT-i-2: {count2}\nIVT-i-3: {count3}\nIVT-i-4: {count4}")
    with open(f"./data/{name}Validation.txt", "w") as outfile:
        count1, count2, count3, count4 = 0,0,0,0
        for rid in validation:
            outfile.writelines(rid[0]+"\n")
            match rid[1]:
                case "IVT-i-1": count1+=1
                case "IVT-i-2": count2+=1
                case "IVT-i-3": count3+=1
                case "IVT-i-4": count4+=1
        print(f"Validation dataset:\nIVT-i-1: {count1}\nIVT-i-2: {count2}\nIVT-i-3: {count3}\nIVT-i-4: {count4}\n")


split_data(0.8, 0.1, transcripts0, "group0") #Train: 80%, Test: 10%, Validation: 10%
split_data(0.8, 0.1, transcripts1, "group1")
