import sys, time, datetime

filename = '{0}.txt'.format(str(time.time()))
start = time.time()

def printNowToFile(inp):
    inp2 = datetime.datetime.now()
    print(inp, inp2)
    end = time.time()
    print('time since start: {0}'.format( str(datetime.timedelta(seconds=(end - start)))))
    original_stdout = sys.stdout
    with open(filename, 'a') as f:
        sys.stdout = f
        print(inp,inp2)
        print('time since start: {0}'.format( str(datetime.timedelta(seconds=(end - start)))))
    sys.stdout = original_stdout

def printToFile(inp):
    print(inp)
    original_stdout = sys.stdout
    with open(filename, 'a') as f:
        sys.stdout = f
        print(inp)
    sys.stdout = original_stdout

