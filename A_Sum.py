if __name__ == "__main__":
    # a = list(map(int,input().split()))
    # b = list(map(int,input().split()))
    # c = list(map(int,input().split()))
    # d = list(map(int,input().split()))
    # e = list(map(int,input().split()))
    # f = list(map(int,input().split()))
    # g = list(map(int,input().split()))
    # h =list(map(int,input().split()))

    # lists =[a,b,c,d,e,f,g,h]
    # for item in lists:
    #     maximum = max(item)
    #     if maximum == sum(item)-maximum:
    #         print("yes")
    #     else:
    #         print("no")
      
    size = int(input())
    for i in range(size):
         a = list(map(int,input().split()))
         maximum = max(a)
         if maximum == sum(a)-maximum:
            print("YES")
         else:
            print("NO")


        