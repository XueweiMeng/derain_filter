import argparse
import re
import os
import shutil


def get_arguments():
    """Parse the arguments from the command line. 

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser( 
            description='Script for the preparation of dehaze dataset')

    parser.add_argument('--clear_folder', default='ITS_v2/clear',
                         help='Folder with dehaze clear datasets')
    parser.add_argument('--hazy_folder', default='ITS_v2/hazy', 
                         help='Folder with dehaze hazy datasets')
    return parser.parse_args()


def checkDir(dirPath): 
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def main():
    args = get_arguments()
    if not os.path.exists(args.clear_folder): 
        raise FileNotFoundError('Folder %s not found!' % args.dataset_folder)
    if not os.path.exists(args.hazy_folder):
        raise FileNotFoundError('Folder %s not found!' % args.dataset_folder)

    norainDir_train = 'defog_dataset/train_datasets/norain'
    rainDir_train   = 'defog_dataset/train_datasets/rain/X2'
    norainDir_test  = 'defog_dataset/test_datasets/norain'
    rainDir_test    = 'defog_dataset/test_datasets/rain/X2'

    checkDir('defog_dataset')
    checkDir('defog_dataset/train_datasets')
    checkDir('defog_dataset/test_datasets')
    checkDir('defog_dataset/train_datasets/rain')
    checkDir('defog_dataset/test_datasets/rain')
    checkDir(norainDir_train)
    checkDir(norainDir_test)
    checkDir(rainDir_train)
    checkDir(rainDir_test)

    # prepare clear files for train and test
    for clear_files in os.listdir(args.clear_folder):
        oldname = os.path.join(args.clear_folder, clear_files)

        for i in range(10):
            j = i + 1
            newfile=clear_files[0:len(clear_files)-4]+'_'+str(j)+'.png'
            if (int)(clear_files.split('.')[0]) <= 1360:
                newfile_path=os.path.join(norainDir_train,newfile)
            else:
                newfile_path=os.path.join(norainDir_test,newfile) 
            shutil.copyfile(oldname,newfile_path)
    

    # prepare hazy files for train and test
    for hazy_files in os.listdir(args.hazy_folder):
        oldname = os.path.join(args.hazy_folder, hazy_files)
        newfile = hazy_files[0: len(hazy_files) - len(hazy_files.split('_')[2]) - 1] + 'x2.png'
        if (int)(hazy_files.split('_')[0]) <= 1360:
            newname = os.path.join(rainDir_train, newfile)
        else:
            newname = os.path.join(rainDir_test, newfile) 
        shutil.copyfile(oldname,newname)

    shutil.rmtree(args.clear_folder)
    shutil.rmtree(args.hazy_folder)

if __name__ == '__main__':
    main()

