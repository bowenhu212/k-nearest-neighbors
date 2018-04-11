import numpy

SIZE = 8
index = 0

def init():
    plate = numpy.chararray((SIZE, SIZE))
    plate[:] = '.'
    return plate
    
def iterationSearch():
    plate = init()
    setLocation(plate, 0)        

def setLocation(plate, col):
    if col >= SIZE:
        global index
        index += 1
        plate[plate == b'x'] = '.'
        print("No. " + str(index) + ":")
        print(" "+ numpy.array_str(numpy.core.defchararray.decode(plate)).replace('[','').replace(']','').replace("'",''))
        print("========================")
        print()
    else:
        for row in range(SIZE):
            if (b'.' not in plate[:,col]):
                return
            elif (plate[row][col] == b'x'):
                continue
            else:
                newPlate = numpy.copy(plate)
                newPlate[row][col] = '*'
                setDisable(newPlate, row, col)
                setLocation(newPlate, col+1)

def setDisable(plate, row, col):
    plate[row:row+1] = 'x'
    if (col >= row):
        numpy.fill_diagonal(plate[row-row:, col - row:], 'x')
    else:
        numpy.fill_diagonal(plate[row-col:, col - col:], 'x')
    if ((SIZE - 1 - col) >= row):    
        numpy.fill_diagonal(numpy.fliplr(plate)[row - row:, (SIZE - 1 - col) - row:], 'x')
    else:
        numpy.fill_diagonal(numpy.fliplr(plate)[row - (SIZE - 1 - col):, (SIZE - 1 - col) - (SIZE - 1 - col):], 'x')
    plate[row,col] = '*'
    
def main():
    iterationSearch()
    
if __name__ == '__main__':
    main()