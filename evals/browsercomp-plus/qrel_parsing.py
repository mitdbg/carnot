from collections import defaultdict
qrel = defaultdict(list)
with open("topics-qrels/qrel_evidence.txt") as f:
    for line in f:
        qid, _, docid, rel = line.split()
        if rel != "0":
            qrel[qid].append(docid)
i = 0
maxLen = 0
for key in qrel:
    if i > 100:
        break
    maxLen = max(maxLen, len(qrel[key]))
    #print(key, len(qrel[key]))
    i += 1
print(maxLen)