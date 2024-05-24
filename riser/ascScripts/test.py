import pysam
samfile = pysam.AlignmentFile("SM47_IVT_NM_RNA004.IVT-i.filtered.bam", "rb")

group0 = ["IVT-i-2", "IVT-i-4"]
group1 = ["IVT-i-1", "IVT-i-3"]
#Counts how many of each IVT there are in the specified bam file.
transcripts0 = []
transcripts1 = []
grouptoreadID = dict()
count1, count2, count3, count4 = 0,0,0,0
for line in samfile:
    transcript = line.reference_name
    match transcript:
        case "IVT-i-1": 
            count1 += 1
        case "IVT-i-2":
            count2 += 1
        case "IVT-i-3":
            count3 += 1
        case "IVT-i-4":
            count4 += 1
        case _:
            raise Exception

print(str(count1) + ", " + str(count2) + ", " + str(count3) + ", " + str(count4))
#535712, 450098, 559554, 400265 => lowest is IVT-i4, with 400265 entries