import os

directory = "D:\Coding\School\img_rec_proj\EmptyData\\Train"
string = "{"
counter = 0
lst = os.listdir(directory)
lst.sort()
for folder in lst:
    
    string += str(counter)+":"+"'"+folder+"', "
    counter += 1

string += "}"
print(string)