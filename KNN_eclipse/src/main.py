SIZE = 8
index = 0


def init():
    plate = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append('.')
        plate.append(row)
    return plate


def clone(plate):
    newPlate = init()
    for i in range(SIZE):
        for j in range(SIZE):
            newPlate[i][j] = plate[i][j]
    return newPlate


def iterationSearch():
    plate = init()
    setLocation(plate, 0)


def setLocation(plate, col):
    if col >= SIZE:
        global index
        index += 1
        trim(plate)
        print("No. " + str(index) + ":")
        printf(plate)
        print("========================")
        print()
    else:
        for row in range(SIZE):
            if (nonSpace(plate, col)):
                return
            elif (plate[row][col] == 'x'):
                continue
            else:
                newPlate = clone(plate)
                newPlate[row][col] = '$'
                setDisable(newPlate, row, col)
                setLocation(newPlate, col + 1)


def setDisable(plate, row, col):
    for i in range(SIZE):
        if (plate[row][i] == '.'):
            plate[row][i] = 'x'
        if (col - row + i < SIZE and col - row + i >= 0 and plate[i][col - row + i] == '.'):
            plate[i][col - row + i] = 'x'
        if (col - (SIZE - 1 - row) + i < SIZE and col - (SIZE - 1 - row) + i >= 0 and plate[SIZE - 1 - i][
            col - (SIZE - 1 - row) + i] == '.'):
            plate[SIZE - 1 - i][col - (SIZE - 1 - row) + i] = 'x'


def nonSpace(p, col):
    for i in range(SIZE):
        if (p[i][col] == '.'):
            return False;
    return True;


def trim(plate):
    for i in range(SIZE):
        for j in range(SIZE):
            if (plate[i][j] == 'x'):
                plate[i][j] = '.'


def printf(p):
    i = 'A'
    j = 1
    for row in p:
        for col in row:
            print(" " + col, end="")
            #if(col == '*'):
            #    print(i + str(j))
            #j = j + 1
        print()
        #i = chr(ord(i)+1)
        #j = 1

def main():
    iterationSearch()


if __name__ == '__main__':
    main()