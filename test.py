import time
import os


def main():
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    time.sleep(60*60)


if __name__ == '__main__':
    main()
