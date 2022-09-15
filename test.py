import time

if __name__ == "__main__":
    print('sleeping')
    test = ['/home/data/32604/2000-2020/tiles/trendimage_32604_150_62.tif', '/home/data/32604/2000-2020/tiles/trendimage_32604_150_67.tif', '/home/data/32604/2000-2020/tiles/trendimage_32604_150_66.tif', '/home/data/32604/2000-2020/tiles/trendimage_32604_150_63.tif',
             '/home/data/32604/2000-2020/tiles/trendimage_32604_150_65.tif', '/home/data/32604/2000-2020/tiles/trendimage_32604_150_64.tif']
    for t in test:
        print(t)
    test.sort()
    print('not sorting')
    for t in test:
        print(t)
    time.sleep(60*60)