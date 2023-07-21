import os
 
path = r'C:\Users\kiran\OneDrive\Desktop\myFile.txt'

if os.path.isfile(path):
    print("File found")
    try: 
        text=open(path).read()
        print(text)
        # text_edit.setPlainText(text)
    except:
        print("Error Occurred")
    

else:
    print("File not found")