import shutil
from concurrent.futures import ThreadPoolExecutor
import os
import argparse

def copy_file(file, dest_folder):
    shutil.copy(file, dest_folder)
    print(f"Copied {file} to {dest_folder}")

def delete_files(directory_path):
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

destinationFolders = [
            '/raid/renuga/juniarto/dataset/local1',
            '/raid/renuga/juniarto/dataset/local2',
            '/raid/renuga/juniarto/dataset/local3',
            '/raid/renuga/juniarto/dataset/global'
            ]

sourceFolders = [
    '/home/renuga_k/juniarto/evyd-shapley-api-server/uploads/667e73e1a177650847d637ef/667e73e3a177650847d637f0',
    '/home/renuga_k/juniarto/evyd-shapley-api-server/uploads/667e73e1a177650847d637ef/667e73fba177650847d637f1',
    '/home/renuga_k/juniarto/evyd-shapley-api-server/uploads/667e73e1a177650847d637ef/667e740ea177650847d637f2',
    '/home/renuga_k/juniarto/evyd-shapley-api-server/uploads/667e73e1a177650847d637ef/global'
]

def batch2():
    localFiles = ['local6.pth.tar', 'local7.pth.tar']
    globalFiles = ['global6.pth.tar', 'global7.pth.tar']
    with ThreadPoolExecutor(max_workers=4) as executor:
        for myLocal, myGlobal in zip(localFiles, globalFiles):
            for (j, (myDestination, mySource)) in enumerate(zip(destinationFolders, sourceFolders)):
                if j == 3:
                    myGlobalFile = os.path.join(mySource, myGlobal)
                    executor.submit(copy_file, myGlobalFile, myDestination)
                else:
                    myLocalFile = os.path.join(mySource, myLocal)
                    #f0/local5
                    #f1/local5
                    #f2/local5
                    executor.submit(copy_file, myLocalFile, myDestination)
                    #local1
                    #local2
                    #local3


def batch1():
    localFiles = ['local3.pth.tar','local4.pth.tar', 'local5.pth.tar']
    globalFiles = ['global3.pth.tar', 'global4.pth.tar', 'global5.pth.tar']
    with ThreadPoolExecutor(max_workers=4) as executor:
        for myLocal, myGlobal in zip(localFiles, globalFiles):
            for (j, (myDestination, mySource)) in enumerate(zip(destinationFolders, sourceFolders)):
                if j == 3:
                    myGlobalFile = os.path.join(mySource, myGlobal)
                    executor.submit(copy_file, myGlobalFile, myDestination)
                else:
                    myLocalFile = os.path.join(mySource, myLocal)
                    #f0/local5
                    #f1/local5
                    #f2/local5
                    executor.submit(copy_file, myLocalFile, myDestination)
                    #local1
                    #local2
                    #local3

def mass_copy():
    localFiles = ['local0.pth.tar', 'local1.pth.tar', 'local2.pth.tar']
    globalFiles = ['global0.pth.tar', 'global1.pth.tar', 'global2.pth.tar']
    with ThreadPoolExecutor(max_workers=4) as executor:
        for myLocal, myGlobal in zip(localFiles, globalFiles):
            for (j, (myDestination, mySource)) in enumerate(zip(destinationFolders, sourceFolders)):
                if j == 3:
                    myGlobalFile = os.path.join(mySource, myGlobal)
                    executor.submit(copy_file, myGlobalFile, myDestination)
                else:
                    myLocalFile = os.path.join(mySource, myLocal)
                    #f0/local5
                    #f1/local5
                    #f2/local5
                    executor.submit(copy_file, myLocalFile, myDestination)
                    #local1
                    #local2
                    #local3

def mass_delete():
    with ThreadPoolExecutor(max_workers=4) as executor:
        for myDestination in destinationFolders:
            executor.submit(delete_files, myDestination)

def main():
    print('Hello')
    parser = argparse.ArgumentParser(description="Simulate Files Uploading")
    parser.add_argument('--command', type=str, help='delete||initiate||first||second')
    

    args = parser.parse_args()
    if args.command == "delete":
        print("Deleting..")
        mass_delete()
    elif args.command == "initiate":
        print("Initiating..")
        mass_copy()
    elif args.command == "first":
        print("Batch 1 ...")
        batch1()
    elif args.command == "second":
        print("Batch 2...")
        batch2()

    

if __name__ == "__main__":
    main()