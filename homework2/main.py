import pandas as pd
import math

df = pd.read_csv('height_data.csv')
dic = df.to_dict()
data = dic['height']
heights = []
for key in data:
    heights.append(data[key])

u1,u2,o1,o2,a1,a2 = 180,170,100,100,0.9,0.1
ep = 0.000001

while True:
    r1=r2=fenzi1=fenzi2=fenzi3=fenzi4=0
    for j in range(len(heights)):
        temp1 = a1 * (1 / o1) * math.exp(-(pow((heights[j] - u1), 2)) / (2 * pow(o1, 2)))
        temp2 = a2 * (1 / o2) * math.exp(-(pow((heights[j] - u2), 2)) / (2 * pow(o2, 2)))
        temp3 = temp1 / (temp1 + temp2)
        temp4 = 1 - temp3
        r1 += temp3
        r2 += temp4
        fenzi1 += temp3 * heights[j];
        fenzi2 += temp4 * heights[j];
        fenzi3 += temp3 * pow((heights[j] - u1), 2);
        fenzi4 += temp4 * pow((heights[j] - u2), 2);
    if abs(u1- fenzi1/r1)<ep and abs(u2- fenzi2/r2)<ep and abs(o1- math.sqrt(fenzi3/r1))<ep \
        and abs(o2- math.sqrt(fenzi4/r2))<ep and abs(a1- r1/len(heights))<ep \
            and abs(a2- r2/len(heights))<ep:
        break;
    u1 = fenzi1 / r1;
    u2 = fenzi2 / r2;
    o1 = math.sqrt(fenzi3 / r1);
    o2 = math.sqrt(fenzi4 / r2);
    a1 = r1 / len(heights);
    a2 = r2 / len(heights);
    print(round(u1,6),round(u2,6),round(o1,6),round(o2,6),round(a1,6),round(a2,6))
