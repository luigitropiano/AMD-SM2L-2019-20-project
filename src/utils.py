import sys, time, datetime
import argparse

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



def check_port(value):
    if not value.isnumeric():
        raise argparse.ArgumentTypeError(f'{value} is not a valid value')
    ivalue = int(value)
    if ivalue < 1024 or ivalue > 65535:
        raise argparse.ArgumentTypeError(f'{value} is not a valid value')
    return ivalue

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host",
                        help = "set url of the spark master",
                        type = str)
    parser.add_argument("-P", "--port",
                        help = "set port of the spark master, default to 7077",
                        type = check_port,
                        default = 7077)
    #parser.add_argument("-V", "--version", help="show program version", action="store_true")
    return parser
