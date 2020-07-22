def multiply(num1: str, num2: str):
    output = []
    nb1 = 0
    nb2 = 0
    for digits1 in range(len(num1)):
        nb1 = 10*nb1 + int(num1[digits1])
    for digits2 in range(len(num2)):
        nb2 = 10*nb2 + int(num2[digits2])
    return str(nb1*nb2)
        
num1 = "1234"
num2 = "456"
print(multiply(num1, num2))