import subprocess, webbrowser, time, urllib.request, urllib.error

command_string = "tensorboard --logdir=summaries"
tensorBoard_uri = r'http://localhost:6006/'

def check_tensorBoard(uri):
    site_up = False
    while not site_up:
        try:
            conn = urllib.request.urlopen(uri)
        except:
            print("Error.. Waiting 10 seconds")
            time.sleep(10)
        else:
            site_up = True
            print(f'{tensorBoard_uri} is up')
    return True

try:
    sub_process_out = subprocess.Popen(command_string)
    if check_tensorBoard(tensorBoard_uri):
        webbrowser.open('http://localhost:6006/') 
except subprocess.CalledProcessError as e:
    print ( "Error" )

