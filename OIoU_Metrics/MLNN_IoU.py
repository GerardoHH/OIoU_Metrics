import ModelUtils as mu

def main():
    mu.enviroment()
    mu.reproducibility()
    #mu.build_MLNN_IoU()
    
    #Select a dataset to clasify
    #mu.classify_DataSet_A()
    
    #mu.classify_DataSet_XOR()
    
    mu.classify_DataSet_Spiral()
    
    
if __name__ == "__main__":
    main()