def Draw(array):
    plt.imshow((array*0.5 + 0.5).permute(1,2,0).detach().to(device = 'cpu'))
    plt.axis('off')
def multi_Draw(Input):
    flag = 1
    for obj in Input:
        plt.subplot(1, 5, flag)
        Draw(obj.to(device = 'cpu'))
        flag += 1
    plt.show()
