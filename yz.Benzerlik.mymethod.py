from similarity_measures import Benzerlik

def main():
    measures = Benzerlik()
    
    x = [5,6,7,9,4]
    y = [1,8,9,2,3]
    
    print(measures.manhattan(x,y))
    
if __name__ == '__main__':
    main()
    
    