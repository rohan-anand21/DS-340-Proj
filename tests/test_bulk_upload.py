import csv
import random
k=50
cols=['reviews.rating','reviews.text']
res=[]
with open('Datafiniti_Hotel_Reviews.csv',newline='',encoding='utf-8') as f:
    r=csv.DictReader(f)
    i=0
    for row in r:
        if cols[0] not in row or cols[1] not in row: 
            continue
        i+=1
        item=[row[cols[0]],row[cols[1]]]
        if len(res)<k:
            res.append(item)
        else:
            j=random.randrange(i)
            if j<k:
                res[j]=item
with open('bulk_upload_test.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(cols)
    w.writerows(res)