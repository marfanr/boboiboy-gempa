from model.loader import ModelLoader
import argparse

def main():
    parser = argparse.ArgumentParser(description="Gempa")

    parser.add_argument("mode", help="train|test|ls",type=str)
    parser.add_argument("--source", help="hdf5", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    
    if args.mode == "ls":
        for i in ModelLoader.list_keys():
            print(i)
        return
    
    if args.model is None:
        raise ValueError("--model is required")
    
    # print(model)
        
    if args.mode == "train":
        if args.source == "hdf5":
            print("using hdf5")
            
        


if __name__ == '__main__':
    main()